import json
import csv
import requests
import argparse
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os

# ------------------------ CONFIG ------------------------

PHI3_URL = "http://localhost:11434/api/generate"
PHI3_MODEL = "phi3:mini"
OPENAI_MODEL = "gpt-4"
llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0.7)

# ------------------------ HELPERS ------------------------

def call_phi3(prompt):
    response = requests.post(
        PHI3_URL,
        json={"model": PHI3_MODEL, "prompt": prompt, "stream": True},
        stream=True
    )
    full = ""
    for line in response.iter_lines():
        if line:
            try:
                full += json.loads(line).get("response", "")
            except json.JSONDecodeError:
                continue
    return full.strip()

def call_openai(prompt, system_msg="You are a CalWORKs report assistant. Focus on service shifts, numerical trends, and outcome changes in CalWORKs program delivery in the county."):
    try:
        response = llm([SystemMessage(content=system_msg), HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"[OpenAI Error] {e}")
        return ""

def call_model(prompt, engine="phi3"):
    if engine == "openai":
        return call_openai(prompt)
    return call_phi3(prompt)

def load_structured_sections(structured_path):
    with open(structured_path, 'r') as f:
        return json.load(f)

def load_geometry_blocks(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_blocks_in_range(pages, start, end):
    blocks = []
    for page in pages:
        if (page["page"] + 1) >= start and (end is None or (page["page"] + 1) <= end):
            for block in page["blocks"]:
                blocks.append({
                    "page": page["page"],
                    "category": block.get("category", ""),
                    "text": block.get("text", "").strip(),
                    "region_poly": block.get("region_poly", None)
                })
    return blocks

# ------------------------ SUMMARIZATION ------------------------

def summarize_part_from_texts(blocks, part_title, engine="phi3"):
    text_combined = "\n".join([b["text"] for b in blocks if b["text"].strip()])
    if not text_combined:
        return ""
    prompt = f"""You are analyzing a section of a CalWORKs county report titled '{part_title}'. 

Summarize the key CalWORKs-related trends, service changes, and quantitative outcomes discussed in the following text. 
Focus especially on any shifts in service usage, access barriers, performance metrics, or population trends.

Report Text:
{text_combined}

Summary:"""
    return call_model(prompt, engine)

# ------------------------ MAIN PIPELINE ------------------------

def summarize_pipeline(structured_path, geometry_path, output_csv, engine="phi3"):
    sections = load_structured_sections(structured_path)
    pages = load_geometry_blocks(geometry_path)

    part_rows = []

    for section in sections:
        section_title = section["section"]

        if "Section 10" in section_title:
            continue

        parts = section.get("parts", [])
        if not parts:
            parts = [{
                "title": section_title,
                "start": section["start_page"],
                "end": section.get("end")
            }]

        for part in parts:
            part_title = part["title"]
            start, end = part.get("start"), part.get("end")
            print(f"Processing: {section_title} → {part_title} (p.{start}-{end})")

            blocks = extract_blocks_in_range(pages, start, end)
            part_summary = summarize_part_from_texts(blocks, part_title, engine)
            part_rows.append([
                section_title,
                part_title,
                start,
                end,
                part_summary
            ])

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "part", "start_page", "end_page", "part_summary"])
        writer.writerows(part_rows)

    print(f"\nSaved summaries to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Page-level summarization using Phi3 or OpenAI with emphasis on CalWORKs quantitative analysis.")
    parser.add_argument('--structured', '-s', required=True, help="Path to structured_sections.json")
    parser.add_argument('--geometry', '-g', required=True, help="Path to page_block_text_with_geometry.json")
    parser.add_argument('--output', '-o', default="./part_summaries.csv", help="Output CSV for part summaries")
    parser.add_argument('--engine', '-e', default="phi3", choices=["phi3", "openai"], help="Summarization engine to use")
    args = parser.parse_args()

    summarize_pipeline(args.structured, args.geometry, args.output, args.engine)

if __name__ == "__main__":
    main()