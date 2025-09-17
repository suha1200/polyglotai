import os
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
INDEX_NAME = "polyglotai-v3"

def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    ix = pc.describe_index(INDEX_NAME)
    print(f"Index: {INDEX_NAME} | dim={ix.dimension} | metric={ix.metric}")
    print("Namespaces & counts:")
    # list_namespaces() is not exposed; use stats via queryable namespaces by describe_index() result
    # Some SDKs expose ix.describe_index_stats() — if available in your version uncomment:
    try:
        stats = pc.Index(INDEX_NAME).describe_index_stats()
        for ns, info in (stats.get("namespaces") or {}).items():
            print(f"  {ns}: {info.get('vector_count', 0)}")
    except Exception as e:
        print("  (Stats endpoint not available in this SDK version)")
        print("  Please check the dashboard for per-namespace counts.")

if __name__ == "__main__":
    main()
