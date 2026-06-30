import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

def read_docx(file_path):
    namespaces = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    }
    try:
        with zipfile.ZipFile(file_path) as docx:
            tree = ET.fromstring(docx.read('word/document.xml'))
            paragraphs = []
            for p in tree.iter(f'{{{namespaces["w"]}}}p'):
                text = ''.join(node.text for node in p.iter(f'{{{namespaces["w"]}}}t') if node.text)
                paragraphs.append(text)
            return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error reading {file_path}: {e}"

def main():
    base_dir = Path(r"d:\redrob\[PUB] India_runs_data_and_ai_challenge")
    for docx_path in base_dir.glob("*.docx"):
        txt_path = docx_path.with_suffix(".txt")
        print(f"Reading {docx_path.name}...")
        text = read_docx(docx_path)
        txt_path.write_text(text, encoding="utf-8")
        print(f"Saved to {txt_path.name}")

if __name__ == "__main__":
    main()
