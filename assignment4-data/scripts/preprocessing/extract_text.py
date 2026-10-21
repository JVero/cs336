from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse
import resiliparse.extract.html2text
import resiliparse.parse.encoding
from pathlib import Path


def extract_text_from_html(html: bytes, strict=False) -> str:
    if strict:
        strictness = "strict"
    else:
        strictness = "ignore"
    try:
        encoding = resiliparse.parse.encoding.detect_encoding(html)
        html_str = resiliparse.parse.encoding.bytes_to_str(html, encoding, errors=strictness)
    except UnicodeDecodeError as e:
        print(f"This encoding is not supported: '{encoding}'", html[:100]) 
        print(e.reason, "!!!")
        raise e
    text: str = resiliparse.extract.html2text.extract_plain_text(html_str)
    return text
    

if __name__ == "__main__":
    in_file_path = Path("./local-shared-data/CC/example.warc")
    with open(in_file_path, 'rb') as f:
        for record in ArchiveIterator(f):
            body_content = record.reader.read()
            text = extract_text_from_html(body_content)
            print(text)
            print("================================================================")