import pdfplumber
import re
import json
from collections import defaultdict

def extract_toc_from_pdf(pdf_path, max_pages=2):
    title_pattern = re.compile(
        r"(Introduction|Demographics|Section \d+\..+?|Part \d+\..+?)[\s.]*?(\d{1,3})$",
        re.IGNORECASE
    )

    raw_toc = []

    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(max_pages, len(pdf.pages))):
            text = pdf.pages[i].extract_text()
            for line in text.splitlines():
                line = line.strip().replace("\xa0", " ").replace("\t", " ")
                match = title_pattern.match(line.strip())
                if match:
                    title = re.sub(r"[.\s]+$", "", match.group(1)).strip()
                    page = int(match.group(2))
                    raw_toc.append((title, page))

    return raw_toc

def group_toc_into_structure(toc_list, total_pages):
    structured = []
    current_section = None

    for title, page in sorted(toc_list, key=lambda x: x[1]):
        title_lower = title.lower()

        if title_lower.startswith("demographics") or title_lower.startswith("introduction"):
            structured.append({
                "section": title,
                "start_page": page,
                "parts": []
            })

        elif title_lower.startswith("section"):
            if current_section:
                structured.append(current_section)
            current_section = {
                "section": title,
                "start_page": page,
                "parts": []
            }

        elif title_lower.startswith("part"):
            if current_section is None:
                current_section = {
                    "section": f"(Unknown Section for {title})",
                    "start_page": page,
                    "parts": []
                }
            current_section["parts"].append({
                "title": title,
                "start": page
            })

    if current_section:
        structured.append(current_section)

    # Build timeline and assign end_page
    timeline = []
    for section in structured:
        timeline.append(("section", section["start_page"], section))
        for part in section["parts"]:
            timeline.append(("part", part["start"], part))

    timeline.sort(key=lambda x: x[1])

    for i in range(len(timeline) - 1):
        curr_type, curr_start, curr_obj = timeline[i]
        next_start = timeline[i + 1][1]
        curr_obj["end"] = next_start - 1

    timeline[-1][2]["end"] = total_pages

    return structured

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract structured TOC (section/parts) from first pages of PDF.")
    parser.add_argument('--pdf', required=True, help="Path to PDF file")
    parser.add_argument('--output', default="structured_sections.json", help="Output JSON path")
    args = parser.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        total_pages = len(pdf.pages)
        toc = extract_toc_from_pdf(args.pdf)
        structured = group_toc_into_structure(toc, total_pages)

    with open(args.output, 'w') as f:
        json.dump(structured, f, indent=2)


    print(f"Saved structured TOC to {args.output}")

if __name__ == "__main__":
    main()