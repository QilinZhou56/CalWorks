# LLM-CalWORKs-
LLM-CalWORKs is a lightweight, local large language model (LLM) application powered by Ollama


It is designed to extract, summarize, and compare county-level assessments and findings from CalWORKs-related documents (e.g., Cal-CSA reports). Using models like phi3:mini or mistral running locally through Ollama, the tool enables:

📄 Section-by-section document summarization

🔍 Identification of county-specific barriers and strategies

🔁 Comparison of “Summary of Findings” across counties

🔒 Privacy-preserving analysis (since no cloud access is needed)

This tool automates interpretation of PDF reports and supports CalWORKs researchers, analysts, and program evaluators by making lengthy county submissions more accessible and comparable.


## 🛠️ Setup Instructions

This tool runs **entirely locally**, no OpenAI API or cloud connection required.

### Requirements

- Python 3.8+
- [Ollama](https://ollama.com/download)
- Model: `phi3:mini`

---

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### In your terminal, start the model by running

```bash
ollama run phi3:mini
```

### On the directory of ```summarizer.py```, run the summarizer 

```bash
python3 summarizer.py -p "path/to/Cal-CSA_Orange.pdf" --prompt "Summarize"
```

### A CSV file will be generated
It includes:

PDF: Source file name

Section: Section title (e.g., "Demographics", "Section 1. Outcomes")

Summary: LLM-generated text (or raw text if Section 10, which is the summary section)

