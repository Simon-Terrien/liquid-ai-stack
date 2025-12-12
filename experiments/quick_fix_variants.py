"""Quick fix to make all variants work with same baseline implementation."""

import sys
sys.path.insert(0, 'liquid-shared-core')
sys.path.insert(0, 'experiments')

from metadata_ablation.variants import BaselineRetriever

# Temporarily make all variants use baseline implementation
from metadata_ablation import variants as v

# Replace other retrievers with baseline for now
v.KeywordsRetriever = BaselineRetriever
v.CategoriesRetriever = BaselineRetriever  
v.TaxonomyRetriever = BaselineRetriever
v.HybridRetriever = BaselineRetriever
v.FullEnhancedRetriever = BaselineRetriever

print("✅ All variants now use baseline implementation")
print("   (This is temporary - specific implementations to be added later)")
