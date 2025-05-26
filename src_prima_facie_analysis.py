import re

def preprocess_script(file_path):
    """Reads Prima Facie script and segments PART ONE and PART TWO, extracts scenes."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Segment PART ONE and PART TWO
    part_one_pattern = r'PART ONE.*?(?=PART TWO|$)'
    part_two_pattern = r'PART TWO.*'
    part_one = re.search(part_one_pattern, text, re.DOTALL)
    part_two = re.search(part_two_pattern, text, re.DOTALL)
    part_one_text = part_one.group(0) if part_one else ""
    part_two_text = part_two.group(0) if part_two else ""
    # Extract scenes (if present)
    scene_pattern = r'Scene \d+.*?(?=Scene \d+|PART TWO|$)'
    scenes = re.findall(scene_pattern, text, re.DOTALL)
    return {
        "full_text": text,
        "part_one": part_one_text,
        "part_two": part_two_text,
        "scenes": scenes
    }