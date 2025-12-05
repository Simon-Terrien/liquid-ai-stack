# rag_runtime/rag_agent.py
"""
RAG Agent using LiquidAI for fast inference.

Features:
- Retrieval-augmented generation
- Tool-based context fetching
- Grounded, citation-aware answers
- Streaming support
"""
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from liquid_shared import (
    LocalLiquidModel,
    LFM_SMALL,
    LFM_MEDIUM,
    CONTEXT_TOKENS,
)

from .tools.retrieval import (
    retrieve_chunks,
    hybrid_search,
)


class RAGResponse(BaseModel):
    """Response from the RAG agent."""
    answer: str = Field(description="The generated answer")
    sources: List[str] = Field(
        default_factory=list,
        description="Source references used"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score (0-1)"
    )
    context_used: int = Field(
        default=0,
        description="Number of context chunks used"
    )


class RAGDependencies(BaseModel):
    """Dependencies for the RAG agent."""
    max_context_chunks: int = 5
    use_hybrid_search: bool = True
    min_relevance_score: float = 0.3


RAG_INSTRUCTIONS = f"""You are a knowledgeable assistant that answers questions based on retrieved document context.

CRITICAL RULES:
1. **Use ONLY the provided context** to answer questions. Do not use external knowledge.
2. **If the context is insufficient**, clearly state: "I don't have enough information in the available documents to answer this question."
3. **Cite sources** when providing information. Reference the source number (e.g., [Source 1]).
4. **Be concise** but complete. Aim for clear, actionable answers.
5. **Acknowledge uncertainty** when the context is ambiguous or incomplete.

RESPONSE FORMAT:
- Start with a direct answer to the question
- Support with relevant details from the context
- Include source citations
- End with any caveats or limitations

CONTEXT BUDGET: Maximum ~{CONTEXT_TOKENS} tokens. Focus on the most relevant information.

When you need information, use the search_context tool to find relevant documents.
"""


def create_rag_agent(
    model: Optional[LocalLiquidModel] = None,
    fast_mode: bool = True,
) -> Agent:
    """
    Create the RAG agent.
    
    Args:
        model: Optional pre-loaded model
        fast_mode: If True, use LFM2-700M; otherwise LFM2-1.2B
        
    Returns:
        Configured Pydantic AI Agent with retrieval tools
    """
    if model is None:
        model_id = LFM_SMALL if fast_mode else LFM_MEDIUM
        model = LocalLiquidModel(
            model_id,
            max_new_tokens=512,
            temperature=0.2,
        )
    
    agent = Agent(
        model=model.get_pydantic_model(),
        output_type=RAGResponse,
        deps_type=RAGDependencies,
        instructions=RAG_INSTRUCTIONS,
    )
    
    # Register the search tool
    @agent.tool
    async def search_context(
        ctx: RunContext[RAGDependencies],
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Search for relevant document chunks.
        
        Args:
            query: What to search for
            top_k: Number of results (default from config)
            
        Returns:
            Formatted context from relevant documents
        """
        k = top_k or ctx.deps.max_context_chunks
        
        if ctx.deps.use_hybrid_search:
            results = hybrid_search(query, top_k=k)
        else:
            results = retrieve_chunks(query, top_k=k)
        
        # Filter by relevance
        filtered = [r for r in results if r.score >= ctx.deps.min_relevance_score]
        
        if not filtered:
            return "No relevant documents found for this query."
        
        # Format results
        formatted = []
        for i, r in enumerate(filtered, 1):
            source = r.metadata.get("source", "unknown")
            section = r.metadata.get("section_title", "")
            
            header = f"[Source {i}: {source}"
            if section:
                header += f" - {section}"
            header += "]"
            
            formatted.append(f"{header}\n{r.text}")
        
        return "\n\n---\n\n".join(formatted)
    
    return agent


# Pre-configured agent instances
_rag_agent_fast: Optional[Agent] = None
_rag_agent_quality: Optional[Agent] = None


def get_rag_agent(fast_mode: bool = True) -> Agent:
    """Get or create a RAG agent."""
    global _rag_agent_fast, _rag_agent_quality
    
    if fast_mode:
        if _rag_agent_fast is None:
            _rag_agent_fast = create_rag_agent(fast_mode=True)
        return _rag_agent_fast
    else:
        if _rag_agent_quality is None:
            _rag_agent_quality = create_rag_agent(fast_mode=False)
        return _rag_agent_quality


async def ask(
    question: str,
    agent: Optional[Agent] = None,
    deps: Optional[RAGDependencies] = None,
) -> RAGResponse:
    """
    Ask a question using RAG.
    
    Args:
        question: The question to answer
        agent: Optional agent instance
        deps: Optional configuration
        
    Returns:
        RAGResponse with answer and sources
    """
    if agent is None:
        agent = get_rag_agent(fast_mode=True)
    
    if deps is None:
        deps = RAGDependencies()
    
    result = await agent.run(question, deps=deps)
    
    return result.output


def ask_sync(
    question: str,
    agent: Optional[Agent] = None,
    deps: Optional[RAGDependencies] = None,
) -> RAGResponse:
    """Synchronous version of ask."""
    if agent is None:
        agent = get_rag_agent(fast_mode=True)
    
    if deps is None:
        deps = RAGDependencies()
    
    result = agent.run_sync(question, deps=deps)
    
    return result.output


# Simple function for direct RAG without agent overhead
def simple_rag(
    question: str,
    top_k: int = 5,
    model: Optional[LocalLiquidModel] = None,
) -> str:
    """
    Simple RAG: retrieve + generate in one call.
    
    For cases where you don't need the full agent infrastructure.
    """
    # Retrieve context
    results = retrieve_chunks(question, top_k=top_k)
    
    if not results:
        return "I don't have any relevant documents to answer this question."
    
    # Build context
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{i}] {r.text}")
    
    context = "\n\n".join(context_parts)
    
    # Generate answer
    if model is None:
        model = LocalLiquidModel(LFM_SMALL, max_new_tokens=256, temperature=0.2)
    
    prompt = f"""Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    return model.generate(prompt)


if __name__ == "__main__":
    # Interactive CLI
    
    print("RAG Agent (LiquidAI)")
    print("Type 'exit' to quit\n")
    
    agent = get_rag_agent(fast_mode=True)
    deps = RAGDependencies()
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                break
            if not question:
                continue
            
            print("\nSearching and generating answer...")
            response = ask_sync(question, agent=agent, deps=deps)
            
            print(f"\n{response.answer}")
            if response.sources:
                print(f"\nSources: {', '.join(response.sources)}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
