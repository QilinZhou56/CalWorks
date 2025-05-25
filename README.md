# 🧾 CalWorks Semantic Search System

A semantic search and question-answering system for exploring California Office of Administrative Resources (CalOAR) and CalWORKs County Self-Assessment (Cal-CSA) documents using vector embeddings, Milvus, and GPT-4.

---

## 🔍 Project Overview

This project allows users to query complex administrative documents—like the CalWORKs CSA—for specific information on:

- Demographics and community profiles
- Housing affordability and poverty indicators
- Eligibility and program services
- Client engagement and partner collaboration
- System barriers and policy recommendations

The system uses OpenAI embeddings + Milvus for vector search and GPT-4 to generate concise, accurate answers.

---

## 💡 Key Features

- **Semantic Search**: Retrieve the most relevant blocks of text from CalOAR documents using vector embeddings (`text-embedding-3-small`).
- **LLM-Enhanced Answers**: Use GPT-4 to synthesize human-readable answers from top matches.
- **Section-Aware Retrieval**: Directly reference document sections like `"Demographic Analysis"` or `"Section 3"`.
- **Gradio UI**: Lightweight frontend for interactive exploration and Q&A.
- **Inference Time Feedback**: Display backend processing time to users for transparency.

## GitHub Folder Structure 
CalWorks/
├── pdfs/
│   ├── (Cal-CSA)Orange.pdf 03-22-40-516.pdf
│   └── Cal-CSA-Inyo-Report-County.pdf
└── pipeline/
    ├── Asset/
    │   ├── [GIS]ca_counties_outcome.csv
    │   ├── calworks_logo.jpeg
    │   └── cdss-logo.png
    ├── Code/
    │   ├── [Development]CalWorks_Pipeline.ipynb
    │   └── [Final]CalWorks_Pipeline.ipynb
    ├── Modules/
    │   ├── build_vectordb.py
    │   ├── preprocess_documents.py
    │   └── web_interface_components.py
    └── Output/
        ├── chunked_sip_csa_output.xlsx
        └── query_log.json
1. pdfs/
Contains the raw Cal-CSA PDFs on the website.

2. pipeline/Asset/
Static assets and GIS outcome CSV.

3. pipeline/Code/
Colab/Notebook versions of data ingestion & search pipeline.

4. pipeline/Modules/
Reusable Python modules for building the vector DB, preprocessing, and web UI components.

5. pipeline/Output/
Results of pipeline runs (chunked outputs, query logs, etc.).
