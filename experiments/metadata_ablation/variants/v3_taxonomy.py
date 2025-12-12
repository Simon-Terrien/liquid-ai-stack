"""V3: Taxonomy retriever - Dense + taxonomy-based query expansion."""

import sys
from typing import List, Set, Dict

sys.path.insert(0, "liquid-shared-core")

from liquid_shared import VectorStore, RetrievalResult, EmbeddingService
from .base import BaseRetriever, RetrieverConfig


class TaxonomyRetriever(BaseRetriever):
    """V3: +Taxonomy - Dense retrieval with hierarchical query expansion."""

    # Domain taxonomy for cybersecurity and AI
    TAXONOMY = {
        # AI/ML Security
        "adversarial": ["attack", "perturbation", "robustness", "evasion", "poisoning"],
        "machine learning": ["ml", "ai", "model", "algorithm", "neural network"],
        "attack": ["adversarial", "exploit", "threat", "vulnerability", "breach"],
        "robustness": ["resilience", "defense", "hardening", "security"],
        "model": ["algorithm", "neural network", "classifier", "predictor"],

        # Data Protection & Privacy
        "gdpr": ["data protection", "privacy", "compliance", "regulation", "eu law"],
        "privacy": ["data protection", "confidentiality", "personal data", "gdpr"],
        "data protection": ["privacy", "security", "gdpr", "compliance"],
        "compliance": ["regulation", "policy", "governance", "standards"],
        "personal data": ["pii", "privacy", "data protection", "sensitive data"],

        # Cyber Insurance
        "insurance": ["coverage", "premium", "underwriting", "risk assessment", "actuarial"],
        "cyber insurance": ["coverage", "risk transfer", "cyber risk", "insurance"],
        "risk assessment": ["risk analysis", "threat assessment", "vulnerability assessment"],
        "actuarial": ["risk modeling", "premium", "underwriting", "statistics"],
        "coverage": ["insurance", "protection", "indemnity", "policy"],

        # Technical Controls
        "monitoring": ["detection", "surveillance", "observation", "tracking"],
        "anomaly detection": ["outlier detection", "behavioral analysis", "intrusion detection"],
        "input validation": ["sanitization", "verification", "filtering", "validation"],
        "defense": ["protection", "security", "mitigation", "countermeasure"],
        "mitigation": ["remediation", "countermeasure", "defense", "protection"],

        # Threat Intelligence
        "threat": ["attack", "risk", "vulnerability", "exploit", "malware"],
        "intelligence": ["information", "knowledge", "insight", "analysis"],
        "vulnerability": ["weakness", "flaw", "bug", "exploit", "cve"],
        "exploit": ["attack", "vulnerability", "threat", "breach"],

        # Risk Management
        "incident response": ["security incident", "breach response", "disaster recovery"],
        "risk": ["threat", "vulnerability", "exposure", "hazard"],
        "cybersecurity": ["security", "information security", "cyber defense"],
        "security": ["protection", "defense", "safeguard", "cybersecurity"],

        # Governance & Policy
        "audit": ["assessment", "review", "evaluation", "compliance check"],
        "governance": ["management", "oversight", "control", "policy"],
        "regulation": ["law", "compliance", "standard", "requirement"],
        "policy": ["guideline", "standard", "procedure", "rule"],
        "framework": ["methodology", "structure", "model", "architecture"],

        # Research & Innovation
        "research": ["study", "investigation", "analysis", "examination"],
        "innovation": ["advancement", "development", "improvement", "progress"],
        "methodology": ["approach", "method", "technique", "framework"],
        "analysis": ["examination", "evaluation", "assessment", "investigation"],
    }

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 10,
        expansion_depth: int = 2,
        max_expansions: int = 10,
    ):
        config = RetrieverConfig(
            name="V3: +Taxonomy",
            description="Dense + taxonomy expansion",
            top_k=top_k,
        )
        super().__init__(vector_store, embedding_service, config)
        self.expansion_depth = expansion_depth  # How many levels to expand
        self.max_expansions = max_expansions    # Max expansion terms

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for phrase matching."""
        import re
        text = text.lower()
        # Extract multi-word phrases and single words
        tokens = []
        # Try to match multi-word terms first
        for term in self.TAXONOMY.keys():
            if term in text:
                tokens.append(term)
        # Add individual words
        words = re.findall(r'\b\w+\b', text)
        tokens.extend(words)
        return list(set(tokens))

    def _expand_query_with_taxonomy(self, query: str) -> List[str]:
        """Expand query with related taxonomy terms using hierarchical expansion.

        Process:
        1. Extract key terms from query
        2. Find related terms in taxonomy (1st level)
        3. Optionally expand related terms (2nd level)
        4. Return unique expansion terms
        """
        query_terms = self._tokenize(query)
        expansion_terms = set()

        # Level 1: Direct expansions
        for term in query_terms:
            if term in self.TAXONOMY:
                expansion_terms.update(self.TAXONOMY[term][:5])  # Top 5 related terms

        # Level 2: Expand the expansion terms (if depth > 1)
        if self.expansion_depth > 1:
            level1_terms = list(expansion_terms)
            for term in level1_terms:
                if term in self.TAXONOMY:
                    expansion_terms.update(self.TAXONOMY[term][:2])  # Top 2 per term

        # Remove query terms from expansions (don't duplicate)
        expansion_terms -= set(query_terms)

        # Limit total expansions
        expansion_list = list(expansion_terms)[:self.max_expansions]

        return expansion_list

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Dense search with taxonomy-based query expansion.

        Process:
        1. Extract key terms from query
        2. Expand with related taxonomy terms
        3. Create expanded query
        4. Dense retrieval on expanded query
        """
        # Expand query
        expansion_terms = self._expand_query_with_taxonomy(query)

        # Create expanded query
        if expansion_terms:
            expanded_query = f"{query} {' '.join(expansion_terms)}"
        else:
            expanded_query = query  # Fallback if no expansions

        # Dense retrieval with expanded query
        query_embedding = self._query_to_embedding(expanded_query)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.config.top_k
        )
        return self._results_to_retrieval_results(results)
