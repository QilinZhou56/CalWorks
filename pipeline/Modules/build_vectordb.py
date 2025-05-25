import os
import re
import shutil
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configuration
# Change Path accordingly to the directories where you placed the text chunk excel file and where you want to store the
# vector database accordingly
XLSX_PATH   = "/content/drive/MyDrive/LLM/CalWorks/Vector Database/Output/chunked_sip_csa_output.xlsx"
PERSIST_DIR = "/content/drive/MyDrive/LLM/CalWorks/Vector Database/Output/chroma_sip_csa_db"
COLLECTION  = "sip_csa_chunks"
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert OPENAI_API_KEY, "OPENAI_API_KEY not set in environment"

# Normalize Text
def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"‐|–|—", "-", text)
    text = re.sub(r"“|”|\"|''", '"', text)
    text = re.sub(r"’|‘|`", "'", text)
    return text

# Load Excel Data
df = pd.read_excel(XLSX_PATH).dropna(subset=["text"])
df["chunk_id"] = df.apply(
    lambda row: f"{row['county'].replace(' ', '')}_{row['report_type'].replace('-', '')}_{row['section'].replace(':', '').replace('.', '').replace(' ', '')}_chunk{row.name}",
    axis=1
)
df["text"] = df["text"].apply(normalize_text)
df["section"] = df["section"].astype(str).apply(normalize_text)

# Initialize Chroma Client
chroma_client = chromadb.PersistentClient(
    path=PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False)
)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_MODEL
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}
)

print(f"📦 Collection ready: {COLLECTION}")
print(f"🔎 Initial document count: {collection.count()}")

# Prepare Lists
docs = [
    f"County: {row['county']}\nSection: {row['section']}\n\n{row['text']}\n\n[End of Section: {row['section']} – County: {row['county']}]"
    for _, row in df.iterrows()
]
ids   = df["chunk_id"].astype(str).tolist()
metas = df[["county", "report_type", "section", "page"]].to_dict(orient="records")

# Upload with Debug Logs
BATCH_SIZE = 100
for i in range(0, len(docs), BATCH_SIZE):
    end = min(i + BATCH_SIZE, len(docs))
    collection.add(
        documents=docs[i:end],
        ids=ids[i:end],
        metadatas=metas[i:end]
    )
    print(f"Uploaded batch {i//BATCH_SIZE + 1}: {end - i} items")

# Post-upload verification
print(f"Final document count in collection: {collection.count()}")