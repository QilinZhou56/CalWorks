# Revised QA system with county-wise comparison and map-reduce summarization
import os
import getpass
import gradio as gr
import json
import ast
from datetime import datetime
from collections import defaultdict
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document as LCDocument
from langchain_community.utilities import SerpAPIWrapper
from pipeline.Modules.web_interface_components import plot_outcome_map, extract_text_from_file

# Enter authentication keys
OPENAI_API_KEY = getpass.getpass("🔑 Enter your OpenAI API key: ")
SERP_API_KEY = getpass.getpass("🔑 Enter your SERP API key: ")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_API_KEY"] = SERP_API_KEY

# Config
PERSIST_DIR = "/content/drive/MyDrive/LLM/CalWorks/Vector Database/Output/chroma_sip_csa_db"
COLLECTION_NAME  = "sip_csa_chunks"
QUERY_LOG_PATH = "/content/drive/MyDrive/LLM/CalWorks/Vector Database/Output/query_log.json"
TOP_K_DEFAULT = 5
CALIFORNIA_COUNTIES = ["Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa", "Contra Costa",
    "Del Norte", "El Dorado", "Fresno", "Glenn", "Humboldt", "Imperial", "Inyo", "Kern", "Kings", "Lake", "Lassen",
    "Los Angeles", "Madera", "Marin", "Mariposa", "Mendocino", "Merced", "Modoc", "Mono", "Monterey", "Napa",
    "Nevada", "Orange", "Placer", "Plumas", "Riverside", "Sacramento", "San Benito", "San Bernardino",
    "San Diego", "San Francisco", "San Joaquin", "San Luis Obispo", "San Mateo", "Santa Barbara", "Santa Clara",
    "Santa Cruz", "Shasta", "Sierra", "Siskiyou", "Solano", "Sonoma", "Stanislaus", "Sutter", "Tehama", "Trinity",
    "Tulare", "Tuolumne", "Ventura", "Yolo", "Yuba"]
MAX_CHAR_LIMIT = 80000

def extract_counties_from_query(query):
    if any(w in query.lower() for w in ["all counties", "county-wise", "statewide", "compare counties"]):
        return set(CALIFORNIA_COUNTIES)
    return {c for c in CALIFORNIA_COUNTIES if c.lower() in query.lower()}

def load_query_log():
    try:
        with open(QUERY_LOG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_query_log(counter):
    with open(QUERY_LOG_PATH, "w") as f:
        json.dump(counter, f, indent=2)

# Embeddings and Retriever
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
openai_ef = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIR, embedding_function=openai_ef)
retriever = vectorstore.as_retriever()
search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

# LLM + Chains
llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o")
summarizer = load_summarize_chain(llm, chain_type="map_reduce")

def summarize_with_map_reduce(docs):
    docs_lc = []
    for i, doc in enumerate(docs):
        ref_id = f"[{i+1}]"
        meta = doc.metadata
        county = meta.get("county", "Unknown")
        section = meta.get("section", "Unknown Section")
        page = meta.get("page", "?")
        report_type = meta.get("report_type", "Unknown")
        header = f"{ref_id} 📍 {county} | {report_type} | Section: {section} | Page {page}"
        content = f"{header}\n{doc.page_content.strip()}"
        docs_lc.append(LCDocument(page_content=content))
    return summarizer.run(docs_lc)

qa_prompt_template = PromptTemplate(
    input_variables=["context", "question", "external", "user_context"],
    template="""
You are analyzing CalWORKs CSA and SIP county reports.

You may use the following sources:
- Internal report excerpts (Context)
- Optional external web information (External Info)
- User-supplied input (User Context)

Answer the question based on the provided information. If relevant, incorporate insights from external or user context.
If no reliable answer is available, say so directly without guessing.

Context:
{context}

External Info:
{external}

User Context:
{user_context}

Question: {question}

Answer:
"""
)

qa_chain = LLMChain(llm=llm, prompt=qa_prompt_template)
query_log = load_query_log()

def analyze_uploaded_file(uploaded_file, query=""):
    if uploaded_file is None:
        return "❌ Please upload a file."

    file_text = extract_text_from_file(uploaded_file)
    if not file_text.strip():
        return "❌ Could not extract meaningful text from the file."

    file_text = file_text[:MAX_CHAR_LIMIT]
    if not query.strip():
        query = "Summarize the key points or findings in the uploaded document."

    try:
        response = qa_chain.invoke({
            "context": clean_unicode(file_text),
            "question": clean_unicode(query),
            "external": "",
            "user_context": "This is a file upload with optional query input."
        })["text"]
        return response.strip()
    except Exception as e:
        return f"❌ Failed to process file with LLM: {str(e)}"


def clean_unicode(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8")

def query_interface(query, external_query, top_k, use_external, force_all_counties=False):
    query = query.strip()
    external_query = external_query.strip()
    if not query:
        return "Please enter a question.", get_top_queries(), ""

    mentioned = extract_counties_from_query(query)
    docs = retriever.get_relevant_documents(query, k=top_k * 2)

    if mentioned and not force_all_counties:
        docs = [doc for doc in docs if doc.metadata.get("county", "").title() in mentioned][:top_k * 2]
    else:
        docs = docs[:top_k]

    if not docs:
        return "❌ No relevant documents found.", get_top_queries(), ""

    summarized = summarize_with_map_reduce(docs)
    user_context = f"Query targets: {', '.join(sorted(set(doc.metadata.get('county', 'Unknown') for doc in docs)))}"

    external_info = ""

    if use_external and external_query:
        try:
            raw = search_tool.run(external_query)

            # Convert raw string to list if it looks like one
            if isinstance(raw, str) and raw.strip().startswith("["):
                try:
                    raw = ast.literal_eval(raw)
                except Exception:
                    pass  # If parsing fails, fallback to treating it as raw string

            if isinstance(raw, list):
                cleaned = [
                    clean_unicode(s).strip(" '\"\n")
                    for s in raw if isinstance(s, str) and s.strip()
                ]
                external_info = "\n".join(f"• {s}" for s in cleaned[:20])
            else:
                external_info = clean_unicode(str(raw))[:8000].strip()

        except Exception as e:
            external_info = f"External search failed: {e}"


    try:
        response = qa_chain.invoke({
            "context": clean_unicode(summarized),
            "question": clean_unicode(query),
            "external": external_info,
            "user_context": user_context
        })["text"]
    except Exception as e:
        response = f"❌ QA failed: {str(e)}"

    timestamp = datetime.now().isoformat()
    query_log[timestamp] = {
        "query": query,
        "external_query": external_query,
        "counties": sorted(list(mentioned)),
        "used_external": use_external,
        "is_countywise": force_all_counties or len(mentioned) > 3,
        "answer_preview": response[:200]
    }
    save_query_log(query_log)

    excerpts = "\n\n---\n\n".join([
        f"[{i+1}] 📍 {doc.metadata.get('county', 'Unknown')} | {doc.metadata.get('report_type', 'Unknown')} | Section: {doc.metadata.get('section', 'Unknown')} | Page {doc.metadata.get('page', '?')}\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs)
    ])

    return response.strip() + "\n\n📚 Used Excerpts:\n\n" + excerpts, get_top_queries(), external_info.strip()


def get_top_queries(n=20):
    freqs = defaultdict(int)
    for entry in query_log.values():
        freqs[entry["query"]] += 1
    sorted_qs = sorted(freqs.items(), key=lambda x: -x[1])
    return "\n".join([f"{i+1}. {q} — {count}x" for i, (q, count) in enumerate(sorted_qs[:n])]) or "No queries yet."

# Gradio UI
def build_ui():
    with gr.Blocks() as demo:
        gr.Image("/content/drive/MyDrive/LLM/CalWorks/Vector Database/Asset/cdss-logo.png", show_label=False, container=False, width=500)
        gr.Markdown("### 🐻🌲 CalWORKs County QA System")
        gr.Markdown("""📌 **Prompt Tips for County-Wise and Single-County Questions**

You can ask:
- *What is Santa Clara doing to reduce exits and re-entries?*
- *Compare child care availability across counties.*
- *Which counties cite transportation as a major barrier?*

Use phrases like **"county-wise"**, **"statewide"**, or **specific county names** to guide the system.
Avoid vague yes/no questions — ask for comparisons, summaries, or trends.
Include **keywords like** "compare counties", "county-wise", or "statewide" to ensure multi-county retrieval.
""")

        with gr.Row():
            with gr.Column(scale=3):
                qbox = gr.Textbox(label="Ask a Question", placeholder="e.g., Which counties improved work participation?")
                extbox = gr.Textbox(label="External Search Query")
                topk = gr.Slider(1, 50, value=TOP_K_DEFAULT, label="Top k chunks")
                extflag = gr.Checkbox(label="Include Web Search", value=False)
                go = gr.Button("Answer")
                answer = gr.Textbox(label="Answer", lines=10)
                extout = gr.Textbox(label="External Info", lines=10, max_lines=20, show_copy_button=True)
            with gr.Column(scale=3):
                gr.Markdown("### 📎 Upload a File")
                file_upload = gr.File(
                    label="Upload File (PDF, Word, Excel, Image)",
                    file_types=[".pdf", ".docx", ".xlsx", ".xls", ".csv", ".jpg", ".png"],
                    type="filepath"
                )
                query_input = gr.Textbox(label="Optional Question", placeholder="Leave blank to summarize key findings")
                analyze_button = gr.Button("Analyze File")
                file_output = gr.Textbox(label="LLM Summary", lines=10, max_lines=20)
            with gr.Column(scale=1):
                freq = gr.Textbox(label="Top Queries", value=get_top_queries(), lines=20)

        gr.Markdown("### 🗺️ Map of CalWORKs Outcome Metrics by County")
        with gr.Row():
            with gr.Column(scale=1):
                metric_dropdown = gr.Dropdown(
                    choices=OUTCOME_METRICS,
                    value=OUTCOME_METRICS[0],
                    label="Select Outcome Metric"
                )
                county_dropdown = gr.Dropdown(
                    choices=COUNTIES,
                    multiselect=True,
                    label="Select Counties (optional)"
                )
            with gr.Column(scale=4):
                map_plot = gr.Plot(label="Classification by County")

        go.click(query_interface, [qbox, extbox, topk, extflag], [answer, freq, extout])
        analyze_button.click(analyze_uploaded_file, inputs=[file_upload, query_input], outputs=[file_output])

        # Connect dropdowns to the plot callback
        demo.load(plot_outcome_map, inputs=[metric_dropdown, county_dropdown], outputs=[map_plot])
        metric_dropdown.change(plot_outcome_map, inputs=[metric_dropdown, county_dropdown], outputs=[map_plot])
        county_dropdown.change(plot_outcome_map, inputs=[metric_dropdown, county_dropdown], outputs=[map_plot])

        gr.Image("/content/drive/MyDrive/LLM/CalWorks/Vector Database/Asset/calworks_logo.jpeg", show_label=False, container=False, width=1600)

    return demo

demo = build_ui()
demo.launch(debug=True, share=True)
