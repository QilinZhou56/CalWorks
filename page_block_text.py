from collections import defaultdict
import json
import argparse

def load_and_flatten_json(json_path):
    with open(json_path, 'r') as f:
        pages = json.load(f)

    flat_blocks = []
    for page in pages:
        page_num = page.get("page")
        for region in page.get("information", []):
            cat = region.get("category_name")
            region_poly = region.get("region_poly", None)

            region_text = []
            for item in region.get("text_list", []):
                region_text.extend(item.get("content", []))

            content = " ".join(region_text).strip()

            if content:
                flat_blocks.append({
                    "page": page_num,
                    "category": cat,
                    "region_poly": region_poly,
                    "text": content
                })
    return flat_blocks

def group_blocks_by_page(flat_blocks):
    page_map = defaultdict(list)
    for block in flat_blocks:
        page_map[block["page"]].append({
            "category": block["category"],
            "region_poly": block["region_poly"],
            "text": block["text"]
        })

    grouped = []
    for page, blocks in sorted(page_map.items()):
        grouped.append({
            "page": page,
            "blocks": blocks
        })

    return grouped


def main():
    parser = argparse.ArgumentParser(description="Flatten OCR JSON and group by page with category and region info.")
    parser.add_argument('--input', '-i', required=True, help="Path to input OCR JSON file")
    parser.add_argument('--output', '-o', default="page_block_text_with_geometry.json", help="Output JSON file path")
    args = parser.parse_args()

    # Step 1: Flatten all text blocks
    flat_blocks = load_and_flatten_json(args.input)

    # Step 2: Group blocks by page
    grouped_pages = group_blocks_by_page(flat_blocks)

    # Step 3: Save the result
    with open(args.output, 'w') as f:
        json.dump(grouped_pages, f, indent=2)

    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()