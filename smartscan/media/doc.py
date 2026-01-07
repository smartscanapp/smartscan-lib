from smartscan.errors import SmartScanError, ErrorCode
from smartscan.constants import SupportedFileTypes
from smartscan.utils import read_text_file
from urllib.parse import urlparse
import requests

def chunk_text(s: str, tokenizer_max_length: int, limit: int | None = None):
    max_chunks = len(s) // 4 * tokenizer_max_length
    n_chunks = limit if limit else max_chunks
    chunks = []
    start = 0

    while len(chunks) < n_chunks:
        end = start + tokenizer_max_length
        if end >= len(s):
            chunk = s[start:]
        else:
            space_index = s.rfind(" ", start, end)
            if space_index == -1: 
                space_index = end
            chunk = s[start:space_index]
            end = space_index
        if not chunk:
            break
        chunks.append(chunk)
        start = end + 1

    return chunks


def doc_source_to_text_chunks(source: str, tokenizer_max_length: int = 128, max_chunks: int | None = None) -> list[str]:
    if source.startswith(("http://", "https://")):
        path = urlparse(source).path.lower()

        if not path.endswith(SupportedFileTypes.TEXT):
            raise SmartScanError("Unsupported file type", code=ErrorCode.UNSUPPORTED_FILE_TYPE, details=f"Supported file types: {SupportedFileTypes.TEXT}")

        resp = requests.get(source)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if not (content_type.startswith("text/") or "application/json" in content_type):
            raise SmartScanError("Unsupported content type", code=ErrorCode.UNSUPPORTED_FILE_TYPE, details=f"Content-Type: {content_type}")

        raw_text = resp.text
        return chunk_text(raw_text, tokenizer_max_length, max_chunks)
    
    elif source.endswith(SupportedFileTypes.TEXT):
        raw_text = read_text_file(source)
        chunks = chunk_text(source, tokenizer_max_length, max_chunks)
        return chunks
    else:
        raise SmartScanError("Unsupported file type", code=ErrorCode.UNSUPPORTED_FILE_TYPE, details=f"Supported file types: {SupportedFileTypes.TEXT}")
