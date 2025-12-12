#!/usr/bin/env python3
"""Test metadata agent with keywords extraction."""

import asyncio
from etl_pipeline.agents.metadata_agent import extract_metadata

# Test chunk about AI security
test_chunk = """
Machine learning models are vulnerable to adversarial attacks where small perturbations
to input data can cause misclassification. These attacks pose significant security risks
in production AI systems. Organizations must implement robust defenses including adversarial
training, input validation, and model monitoring to protect against these threats.
The GDPR requires that AI systems processing personal data maintain appropriate security measures.
"""

async def test_keywords():
    print("Testing enhanced metadata agent with keywords extraction...")
    print("=" * 70)

    chunks = await extract_metadata(
        chunks=[test_chunk],
        source_path="test.pdf"
    )

    if chunks:
        chunk = chunks[0]
        print(f"\n✅ Metadata extracted successfully!\n")
        print(f"Section Title: {chunk.section_title}")
        print(f"Tags: {chunk.tags}")
        print(f"Entities: {chunk.entities}")
        print(f"Keywords: {chunk.keywords}")  # New field!
        print(f"Importance: {chunk.importance}/10")

        # Verify keywords were extracted
        if chunk.keywords and len(chunk.keywords) >= 5:
            print(f"\n✅ Keywords extraction working! ({len(chunk.keywords)} keywords found)")
        else:
            print(f"\n⚠️ Warning: Expected 5-10 keywords, got {len(chunk.keywords) if chunk.keywords else 0}")
    else:
        print("❌ No chunks returned")

if __name__ == "__main__":
    asyncio.run(test_keywords())
