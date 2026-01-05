def chunk_text(s: str, tokenizer_max_length: int, limit: int = 10):
    max_chunks = len(s) // 4 * tokenizer_max_length
    n_chunks = min(limit, max_chunks)
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