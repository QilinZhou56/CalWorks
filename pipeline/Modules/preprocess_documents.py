import os
import re
import fitz  # PyMuPDF
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
PDF_DIRECTORY = "/content/drive/MyDrive/LLM/CalWorks/Vector Database/PDFs"  # Set this to your PDF folder path

county_names = [
    "Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa", "Contra Costa",
    "Del Norte", "El Dorado", "Fresno", "Glenn", "Humboldt", "Imperial", "Inyo",
    "Kern", "Kings", "Lake", "Lassen", "Los Angeles", "Madera", "Marin", "Mariposa",
    "Mendocino", "Merced", "Modoc", "Mono", "Monterey", "Napa", "Nevada", "Orange",
    "Placer", "Plumas", "Riverside", "Sacramento", "San Benito", "San Bernardino",
    "San Diego", "San Francisco", "San Joaquin", "San Luis Obispo", "San Mateo",
    "Santa Barbara", "Santa Clara", "Santa Cruz", "Shasta", "Sierra", "Siskiyou",
    "Solano", "Sonoma", "Stanislaus", "Sutter", "Tehama", "Trinity", "Tulare",
    "Tuolumne", "Ventura", "Yolo", "Yuba"
]

# Aliases for counties based on common filename patterns
alias_map = {
    "icdss": "Imperial",
    "lacdss": "Los Angeles",
    "ocss": "Orange",
    "scc": "Santa Clara",
    "sbcs": "San Bernardino",
    # Add more aliases as needed
}

def infer_metadata_from_filename(filename):
    # Clean and normalize
    name_clean = filename.lower().replace("_", " ").replace("-", " ").replace(".pdf", "")
    name_compressed = re.sub(r"\s+", "", name_clean)

    # Report type logic
    if "sip" in name_clean or "system improvement" in name_clean:
        report_type = "Cal-SIP"
    elif (
        "csa" in name_clean
        or "self-assessment" in name_clean
        or "calworks self-assessment" in name_clean
        or "county self-assessment" in name_clean
        or "fatal flaw" in name_clean
    ):
        report_type = "Cal-CSA"
    else:
        report_type = "Unknown"

    # Try full county match
    normalized_counties = [c.lower().replace(" ", "") for c in county_names]
    county = "Unknown"
    for orig, compressed in zip(county_names, normalized_counties):
        if compressed in name_compressed:
            county = orig
            break

    # If not found, try alias map
    if county == "Unknown":
        for alias, full_name in alias_map.items():
            if alias in name_compressed:
                county = full_name
                break

    return {
        "file": filename,
        "county": county,
        "report_type": report_type
    }


# A Table of Contents is needed
def extract_sections_via_toc(pdf_path, county, report_type, toc_max_pages=5):
    import fitz
    import re

    doc = fitz.open(pdf_path)
    max_pages = len(doc)
    section_entries = []

    toc_lines = []
    for i in range(min(toc_max_pages, len(doc))):
        text = doc[i].get_text()
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            # Match: Section headers OR named sections like "Introduction", "Executive Summary"
            match = re.match(
                r"^((Section\s+\d+(\.\d+)?[.:]?\s+.+?)|(?:Introduction|Executive Summary|Demographics))\s+\.{3,}\s+(\d{1,3})$",
                line,
                re.IGNORECASE
            )
            if match:
                title = match.group(1).strip()
                page = int(match.group(4))
                toc_lines.append((title, page))

    # Construct section page ranges
    for i, (title, start_page) in enumerate(toc_lines):
        end_page = toc_lines[i + 1][1] - 1 if i + 1 < len(toc_lines) else max_pages
        section_entries.append({
            "county": county,
            "report_type": report_type,
            "section_header": title,
            "start_page": start_page,
            "end_page": end_page
        })

    # Extract section text
    for section in section_entries:
        start = max(0, section["start_page"] - 1)
        end = min(section["end_page"], max_pages)
        text = "".join(doc[p].get_text() for p in range(start, end))
        section["text"] = text.strip()

    return section_entries


def chunk_sections(sections, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunked = []
    for sec in sections:
        if not sec["text"].strip():
            continue
        splits = splitter.split_text(sec["text"])
        for i, chunk in enumerate(splits):
            chunked.append({
                "county": sec["county"],
                "report_type": sec["report_type"],
                "section": sec["section_header"],
                "page": sec["start_page"],
                "chunk_id": f"{sec['section_header']}_chunk{i}",
                "text": chunk
            })
    return chunked

def main():
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
    all_sections = []

    for file in pdf_files:
        meta = infer_metadata_from_filename(file)
        if meta["county"] == "Unknown" or meta["report_type"] == "Unknown":
            print(f"Skipping: {file} (missing county or type)")
            continue
        try:
            path = os.path.join(PDF_DIRECTORY, file)
            sections = extract_sections_via_toc(path, meta["county"], meta["report_type"])

            if len(sections) == 0:
                print(f"No TOC sections found for: {file}")
            else:
                print(f"Found {len(sections)} sections in {file}")
                all_sections.extend(sections)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    chunks = chunk_sections(all_sections)
    print(f"\n{len(all_sections)} sections to {len(chunks)} chunks")

    df = pd.DataFrame(chunks)
    df.to_excel("/content/drive/MyDrive/LLM/CalWorks/Vector Database/Output/chunked_sip_csa_output.xlsx", index=False)
    print("Saved to chunked_sip_csa_output.csv")

if __name__ == "__main__":
    main()