import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

BASE   = Path(__file__).parent.parent.parent.parent
SHARED = BASE / "shared"
DB_PATH = str(BASE / "task_b" / "vectordb")

def build_index():
    print("Loading businesses...")
    df = pd.read_csv(SHARED / "businesses.csv")
    df = df.dropna(subset=["categories", "name", "city"])
    df = df.fillna("")
    print(f"   {len(df):,} businesses loaded")

    print("Loading embedding model (first time downloads ~90MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("   Model loaded")

    print("Building documents for embedding...")
    def build_doc(row):
        return (
            f"{row['name']} is a business in {row['city']}, {row['state']}. "
            f"Categories: {row['categories']}. "
            f"Average rating: {row['stars']} stars from {row['review_count']} reviews."
        )
    df["document"] = df.apply(build_doc, axis=1)

    print("Embedding businesses in batches...")
    docs      = df["document"].tolist()
    ids       = df["business_id"].tolist()
    metadatas = df[["name","city","state","categories","stars","review_count"]].to_dict("records")
    # clean metadata — all values must be str/int/float
    for m in metadatas:
        for k, v in m.items():
            if not isinstance(v, (str, int, float, bool)):
                m[k] = str(v)

    print("Connecting to ChromaDB...")
    os.makedirs(DB_PATH, exist_ok=True)
    client     = chromadb.PersistentClient(path=DB_PATH)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("businesses")
        print("   Cleared existing collection")
    except:
        pass

    collection = client.create_collection(
        name="businesses",
        metadata={"hnsw:space": "cosine"}
    )

    # Embed and insert in batches of 1000
    BATCH = 1000
    total = len(docs)
    for i in range(0, total, BATCH):
        batch_docs  = docs[i:i+BATCH]
        batch_ids   = ids[i:i+BATCH]
        batch_meta  = metadatas[i:i+BATCH]
        embeddings  = model.encode(batch_docs, show_progress_bar=False).tolist()
        collection.add(
            documents  = batch_docs,
            embeddings = embeddings,
            ids        = batch_ids,
            metadatas  = batch_meta
        )
        print(f"   Indexed {min(i+BATCH, total):,} / {total:,}", end="\r")

    print(f"\n   Done. {collection.count():,} businesses indexed")
    return collection

if __name__ == "__main__":
    build_index()
    print("\nVector index built successfully")
