import os
import json
import time
import numpy as np
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from openai import OpenAI

JSON_PATH = "./formatted_outputs/page_block_text_with_geometry.json"
COLLECTION_NAME = "cal_works_clause_index_enhanced"
OPENAI_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
BATCH_SIZE = 10

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Step 1: Load and merge meaningful blocks
with open(JSON_PATH) as f:
    pages = json.load(f)

clauses = []

for page in pages:
    page_num = page["page"] + 1
    blocks = page.get("blocks", [])

    # Merge adjacent "plain text" blocks on the same page
    merged_text = []
    buffer = []
    last_y = None

    for block in blocks:
        category = block.get("category", "")
        text = block.get("text", "").strip()
        bbox = block.get("region_poly", [])
        if not text or category == "page number":
            continue

        y_pos = min(bbox[i+1] for i in range(0, len(bbox), 2)) if bbox else None

        if category != "plain text" or (last_y is not None and y_pos - last_y > 30):
            # Flush buffer
            if buffer:
                full_text = " ".join([b['text'] for b in buffer])
                clauses.append({
                    "text": full_text,
                    "page": page_num,
                    "category": buffer[0]["category"],
                    "bbox": json.dumps(buffer[0]["region_poly"])
                })
                buffer = []
            # Add title or distant block
            clauses.append({
                "text": text,
                "page": page_num,
                "category": category,
                "bbox": json.dumps(bbox)
            })
            last_y = y_pos
        else:
            buffer.append({"text": text, "category": category, "region_poly": bbox})
            last_y = y_pos

    if buffer:
        full_text = " ".join([b['text'] for b in buffer])
        clauses.append({
            "text": full_text,
            "page": page_num,
            "category": buffer[0]["category"],
            "bbox": json.dumps(buffer[0]["region_poly"])
        })

print(f"Extracted {len(clauses)} merged text blocks from OCR.")

# Step 2: Batch embed using OpenAI API
def get_openai_embeddings(text_list):
    all_embeddings = []
    for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc="Embedding text blocks"):
        chunk = text_list[i:i + BATCH_SIZE]
        try:
            response = client.embeddings.create(model=OPENAI_MODEL, input=chunk)
            vectors = [d.embedding for d in response.data]
            all_embeddings.extend(vectors)
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            time.sleep(5)
            continue
    return np.array(all_embeddings)

texts = [c["text"] for c in clauses]
embeddings = get_openai_embeddings(texts)

# Step 3: Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Step 4: Connect to Milvus and define collection
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

if utility.has_collection(COLLECTION_NAME):
    Collection(COLLECTION_NAME).drop()

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4066),
    FieldSchema(name="page", dtype=DataType.INT64),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="bbox", dtype=DataType.VARCHAR, max_length=512),
]

schema = CollectionSchema(fields, description="Merged block-level embeddings for CalOAR OCR text")
collection = Collection(COLLECTION_NAME, schema)

# Step 5: Create index, insert data
collection.create_index("embedding", {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
})
collection.insert([
    embeddings.tolist(),
    [c["text"] for c in clauses],
    [c["page"] for c in clauses],
    [c["category"] for c in clauses],
    [c["bbox"] for c in clauses],
])
collection.load()

