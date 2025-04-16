import requests
import json
import argparse
import pdfplumber
import re
import pandas as pd
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def split_sections(full_text):
    pattern = re.compile(r"(Section\s+\d+\..+|Demographics)", re.IGNORECASE)
    parts = re.split(pattern, full_text)
    sections = {}
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if (i + 1) < len(parts) else ""
        if "summary of findings" in title.lower():
            continue  # Skip unwanted sections
        sections[title] = content
    return sections

def send_prompt(title, content, user_prompt):
    prompt = f"{user_prompt}\n\nSection: {title}\n\n{content}"
    data = {
        "model": "phi3:mini",
        "prompt": prompt,
        "stream": True
    }

    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=data, stream=True)
    full_response = ""

    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                try:
                    json_object = json.loads(line)
                    chunk = json_object.get("response", "")
                    full_response += chunk
                except json.JSONDecodeError:
                    pass
        return full_response.strip()
    else:
        print(f"Error: {response.status_code}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Summarize 9 sections from a PDF and keep raw content of the 10th.")
    parser.add_argument('-p', '--pdf', type=str, required=True, help="Path to the PDF file")
    parser.add_argument('--prompt', type=str, required=True, help="Custom summarization prompt")

    args = parser.parse_args()

    full_text = extract_text_from_pdf(args.pdf)
    if not full_text:
        print("No text extracted.")
        return

    sections = split_sections(full_text)
    if not sections:
        print("No sections detected.")
        return

    pdf_name_only = os.path.splitext(os.path.basename(args.pdf))[0]
    selected_sections = list(sections.items())[:11]  # First 10 valid sections
    summaries = []

    for i, (title, content) in enumerate(selected_sections):
        if i < 10:
            print(f"Summarizing: {title}")
            summary = send_prompt(title, content, args.prompt)
            summaries.append({"PDF": pdf_name_only, "Section": title, "Summary": summary})
        elif i == 10:
            print(f"Saving raw content of: {title}")
            summaries.append({"PDF": pdf_name_only, "Section": title, "Summary": content})

    # Save to CSV
    csv_path = f"{pdf_name_only}_summaries.csv"
    df = pd.DataFrame(summaries, columns=["PDF", "Section", "Summary"])
    df.to_csv(csv_path, index=False)
    print(f"Saved summaries to {csv_path}")

if __name__ == "__main__":
    main()
