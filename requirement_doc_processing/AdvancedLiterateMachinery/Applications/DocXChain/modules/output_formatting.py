#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, re, json, csv, cv2, numpy as np
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load structured JSON
def load_json(path):
    return json.load(open(path))

# Infer image shape from text positions
def infer_image_shape(text_list, margin=50):
    all_coords = np.array([pt for line in text_list for pt in line["position"]])
    xs = all_coords[::2]
    ys = all_coords[1::2]
    width = int(np.max(xs)) + margin
    height = int(np.max(ys)) + margin
    return (height, width)

# Draw and merge close bounding boxes using OpenCV
def draw_and_merge_boxes(text_list, shape, kernel=(12, 4)):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for line in text_list:
        pts = np.array(line["position"]).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(cnt) for cnt in contours]

# Group text lines by bounding boxes
def group_lines(text_list, merged_boxes):
    grouped = {i: [] for i in range(len(merged_boxes))}
    for line in text_list:
        poly = np.array(line["position"]).reshape(4, 2)
        cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
        for i, (x, y, w, h) in enumerate(merged_boxes):
            if x <= cx <= x + w and y <= cy <= y + h:
                content_text = " ".join(line.get("content", []))
                grouped[i].append((content_text, line["position"]))
                break
    return [block for block in grouped.values() if block]

# Build LangChain summarization chain
def build_chain():
    parser = JsonOutputParser()
    prompt = PromptTemplate.from_template(
        "You are a contract analyst. Given this clause:\n\n{clause}\n\n"
        "Extract:\n"
        "- shall: list what must be done\n"
        "- shall_not: list what must not be done\n"
        "- participants: list named roles (e.g. DOE, ATDO, Contractor)\n"
        "- summary: one-paragraph summary of this clause's requirement\n\n"
        "Respond in JSON: {format_instructions}",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    llm = Ollama(model="phi3:mini", temperature=0.2)
    return prompt | llm | parser

# Normalize LLM output items to string
def normalize_list(items):
    return [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False) for item in items]

# Main processing pipeline
def main(input_path="./outputs/manual_structure.json", output_csv="./outputs/sectional_summary.csv"):
    data = load_json(input_path)
    chain = build_chain()
    section_re = re.compile(r"SECTION\s+([A-Z])", re.IGNORECASE)
    clause_re = re.compile(r"([A-Z]\.[0-9]+)[\s\-–]+(.+)", re.IGNORECASE)
    reference_re = re.compile(r"(Clause|Section)\s+[A-Z]\.?[0-9]+", re.IGNORECASE)

    rows = []
    current_section = None

    for page in data:
        page_num = page["page"]
        all_text_lines = []
        for region in page["information"]:
            if region["category_name"] in ["plain text", "header", "title"]:
                all_text_lines.extend(region["text_list"])

        img_shape = infer_image_shape(all_text_lines)
        merged_boxes = draw_and_merge_boxes(all_text_lines, img_shape)
        grouped_blocks = group_lines(all_text_lines, merged_boxes)

        for block in grouped_blocks:
            texts = [b[0] for b in block]
            bbox = block[0][1]
            content = " ".join(texts)

            sec_match = section_re.search(content)
            if sec_match:
                current_section = sec_match.group(1)

            clause_match = clause_re.search(content)
            clause_id = clause_match.group(1) if clause_match else ""
            clause_title = clause_match.group(2).strip() if clause_match else ""

            try:
                result = chain.invoke({"clause": content})
            except Exception as e:
                result = {"shall": [], "shall_not": [], "participants": [], "summary": "", "error": str(e)}

            refs = reference_re.findall(content)
            ref_clauses = [match[0] for match in refs]

            rows.append({
                "section": current_section or "",
                "clause": clause_id,
                "clause_title": clause_title,
                "page": page_num,
                "bounding_box": bbox,
                "text": content,
                "summary": result.get("summary", ""),
                "shall": "; ".join(normalize_list(result.get("shall", []))),
                "shall_not": "; ".join(normalize_list(result.get("shall_not", []))),
                "participants": ", ".join(normalize_list(result.get("participants", []))),
                "references": ", ".join(ref_clauses)
            })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved bounding-box aware section-clause summary to {output_csv}\n")

if __name__ == "__main__":
    main()
