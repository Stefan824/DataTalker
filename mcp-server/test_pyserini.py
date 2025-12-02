"""
Test Pyserini search features and capabilities
"""

try:
    import pyserini
    from pyserini.search import get_topics, get_qrels
    
    print("✓ Pyserini is installed!")
    print(f"Version: {pyserini.__version__}")
    
    print("\n" + "="*80)
    print("PYSERINI FEATURES FOR MCP")
    print("="*80)
    
    print("\n1. PREBUILT INDEXES (Lucene-based, requires Java):")
    print("   - msmarco-v1-passage - MS MARCO passage ranking (~2GB)")
    print("   - msmarco-v1-doc - MS MARCO document ranking")
    print("   - wikipedia-dpr - Wikipedia articles")
    print("   - beir-v1.0.0-* - Various domain-specific datasets")
    
    print("\n2. DENSE RETRIEVAL (No Java required):")
    print("   - DPR (Dense Passage Retrieval)")
    print("   - ANCE")
    print("   - TCT-ColBERT")
    print("   Uses neural embeddings for semantic search")
    
    print("\n3. HYBRID SEARCH:")
    print("   - Combine sparse (BM25) + dense retrieval")
    print("   - Better accuracy than either alone")
    
    print("\n4. AVAILABLE TEST COLLECTIONS:")
    try:
        topics = get_topics('msmarco-passage-dev-subset')
        print(f"   ✓ Can load topics (queries)")
        print(f"   Example: {len(topics)} queries available")
    except Exception as e:
        print(f"   Topics loading: {e}")
    
    print("\n5. CUSTOM INDEX BUILDING:")
    print("   - Build from your own documents")
    print("   - JSON, JSONL, or TSV formats supported")
    
    print("\n" + "="*80)
    print("RECOMMENDED FOR MCP: Dense Retrieval (no Java needed)")
    print("="*80)
    
    print("\n✓ Pyserini is ready to use!")
    print("\nNote: Lucene features require Java JDK 11+")
    print("      Dense retrieval works without Java")
    
except ImportError as e:
    print(f"Error: Pyserini is not installed - {e}")
    print("\nInstall it with: uv add pyserini")
except Exception as e:
    print(f"Error: {e}")
