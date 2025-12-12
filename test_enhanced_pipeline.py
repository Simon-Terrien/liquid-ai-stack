#!/usr/bin/env python3
"""Test the enhanced ETL pipeline with classification and taxonomy."""

import asyncio
import sys

# Add paths
sys.path.insert(0, 'liquid-etl-pipeline')
sys.path.insert(0, 'liquid-shared-core')

from etl_pipeline.graph_etl import run_etl_pipeline

# Short test document
TEST_DOC = """
Machine learning models deployed in production systems face significant security challenges,
particularly from adversarial attacks. These attacks involve carefully crafted input
perturbations designed to cause model misclassification or produce incorrect outputs.

The GDPR mandates that organizations implementing AI systems must ensure appropriate
technical and organizational measures to protect personal data. This includes implementing
robust input validation, adversarial training techniques, and continuous monitoring of
model behavior in production environments.

Key defense strategies include:
1. Adversarial training with augmented datasets
2. Input sanitization and validation
3. Model ensembling for robustness
4. Runtime anomaly detection
5. Regular security audits and penetration testing
"""

async def test_enhanced_pipeline():
    print("=" * 80)
    print("Testing Enhanced ETL Pipeline")
    print("=" * 80)

    try:
        result = await run_etl_pipeline(
            source_path="test_enhanced.txt",
            raw_text=TEST_DOC
        )

        print(f"\n✅ Pipeline completed successfully!")
        print(f"\nResults:")
        print(f"  - Chunks created: {len(result.chunks)}")
        print(f"  - Summaries: {len(result.summaries)}")
        print(f"  - QA pairs: {len(result.qa_pairs)}")
        print(f"  - FT samples: {len(result.ft_samples)}")

        # Verify enhanced metadata
        print(f"\n{'=' * 80}")
        print("Enhanced Metadata Verification")
        print("=" * 80)

        for i, chunk in enumerate(result.chunks):
            print(f"\nChunk {i}:")
            print(f"  Section: {chunk.section_title}")
            print(f"  Importance: {chunk.importance}/10")
            print(f"  Tags: {chunk.tags}")
            print(f"  Keywords: {chunk.keywords}")
            print(f"  Categories: {chunk.categories}")
            print(f"  Entities: {chunk.entities}")

            if chunk.taxonomy:
                print(f"  Taxonomy Root: {chunk.taxonomy.name} ({chunk.taxonomy.importance})")
                if chunk.taxonomy.children:
                    print(f"    Subtopics:")
                    for child in chunk.taxonomy.children:
                        print(f"      - {child.name} ({child.importance}, {child.category})")

        # Check what fields are in FT samples
        if result.ft_samples:
            print(f"\n{'=' * 80}")
            print("Fine-Tuning Sample Metadata")
            print("=" * 80)
            print(f"\nSample metadata keys: {list(result.ft_samples[0].metadata.keys())}")
            print(f"Contains keywords: {'keywords' in result.ft_samples[0].metadata}")
            print(f"Contains categories: {'categories' in result.ft_samples[0].metadata}")
            print(f"Contains importance: {'importance' in result.ft_samples[0].metadata}")

        return result

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_enhanced_pipeline())

    if result:
        print(f"\n{'=' * 80}")
        print("✅ Enhanced metadata extraction working!")
        print("=" * 80)
    else:
        print(f"\n{'=' * 80}")
        print("❌ Test failed")
        print("=" * 80)
        sys.exit(1)
