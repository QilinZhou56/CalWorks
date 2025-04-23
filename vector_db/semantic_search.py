import os
import re
import numpy as np
import gradio as gr
from openai import OpenAI
from pymilvus import connections, Collection

COLLECTION_NAME = "cal_works_clause_index_enhanced"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COLLECTION_NAME)
collection.load()

def embed_query(query_text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query_text]
    )
    vec = np.array(response.data[0].embedding)
    return (vec / np.linalg.norm(vec)).tolist()

def extract_section_number(query):
    match = re.search(r"section\s+(\d+)", query.lower())
    if match:
        return f"Section {match.group(1)}"
    return None

# Retrieve Full Section Text from Milvus 
def retrieve_section(section_title):
    data = collection.query(
        expr="category in ['title', 'plain text']",
        output_fields=["text", "page", "category"]
    )
    data.sort(key=lambda x: (x["page"], 0 if x["category"] == "title" else 1))

    start_idx = next((i for i, item in enumerate(data)
                      if item["category"] == "title" and section_title.lower() in item["text"].lower()), None)
    if start_idx is None:
        return f"Section '{section_title}' not found."

    section_text = []
    for item in data[start_idx + 1:]:
        if item["category"] == "title":
            break
        section_text.append(f"[Page {item['page']}] {item['text']}")

    return f"**{section_title}**\n\n" + "\n\n".join(section_text) if section_text else "⚠️ No content found under section."

# Generate LLM Answer from Retrieved Chunks 
def answer_with_gpt(query, retrieved_texts):
    context = "\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(retrieved_texts)])
    prompt = f"""
You are an expert in analyzing California government contract documents.

Use the excerpts below to answer the question:

EXCERPTS:
{context}

QUESTION:
{query}

ANSWER:
""".strip()

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Main Search Logic 
def semantic_search(query, k=15):
    section_title = extract_section_number(query)
    if section_title:
        return retrieve_section(section_title)

    query_vec = [embed_query(query)]
    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=k,
        output_fields=["text", "page", "category"]
    )

    # Sort by page, then by text length
    sorted_hits = sorted(
        results[0],
        key=lambda hit: (hit.entity.get("page"), len(hit.entity.get("text") or ""))
    )

    top_chunks = [hit.entity.get("text") for hit in sorted_hits]
    answer = answer_with_gpt(query, top_chunks)

    references = "\n\n".join([
        f"**[{i+1}] Page {hit.entity.get('page')} ({hit.entity.get('category')})**:\n{hit.entity.get('text')}"
        for i, hit in enumerate(sorted_hits)
    ])

    return f"📌 **Answer:**\n\n{answer}\n\n---\n\n📄 **Top Retrieved Chunks (Sorted by Page & Length):**\n\n{references}"



# Gradio Interface 
with gr.Blocks(title="CalOAR Document Search") as demo:
    gr.Markdown("# 🧾 CalOAR Document Explorer")
    gr.Markdown("""
Explore CalOAR documents with AI-powered question answering.

Ask about services, demographics, eligibility, or refer to a section like "Demographic Analysis".

### 🧠 Example Questions:
- What are the strengths and needs of Orange County communities?
- How affordable is housing for CalWORKs families?
- What are the trends in employment and income?
- What does the demographic analysis include?
- How were clients and partners engaged in the Cal-CSA?
- What barriers to service access are identified?
- What is the Cal-OAR framework?
""")

    with gr.Row():
        query_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g. What are the support services for child care?"
        )

    status_display = gr.Textbox(value="", label="System Status", interactive=False)

    output_display = gr.Markdown()

    def wrapped_search(query):
        result = semantic_search(query)
        return result, "✅ The LLM assistant has finished interpretation."

    query_input.change(
        fn=wrapped_search,
        inputs=query_input,
        outputs=[output_display, status_display]
    )

demo.launch()

