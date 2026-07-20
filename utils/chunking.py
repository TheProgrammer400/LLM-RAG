import tiktoken
import re

# Use the standard cl100k_base encoding (used by GPT-4 and similar models)
encoding = tiktoken.get_encoding("cl100k_base")


def num_tokens(text: str) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def chunk_text(text, chunk_size=256, overlap=64):
    if not text:
        return []

    # First, split the document into paragraphs
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = num_tokens(para)

        # If adding this paragraph exceeds chunk size and we already have some text
        if current_tokens + para_tokens > chunk_size and current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(chunk_content)

            # Rebuild context for overlap using paragraphs
            overlap_content = []
            overlap_tok_count = 0
            for p in reversed(current_chunk):
                p_tok = num_tokens(p)
                if overlap_tok_count + p_tok + 2 <= overlap:
                    overlap_content.insert(0, p)
                    overlap_tok_count += p_tok + 2
                else:
                    break
            current_chunk = overlap_content
            current_tokens = sum(num_tokens(p) + 2 for p in current_chunk) - 2 if current_chunk else 0

        # If a single paragraph is larger than chunk_size, split it by lines
        if para_tokens > chunk_size:
            lines = para.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line_tokens = num_tokens(line)

                if current_tokens + line_tokens > chunk_size and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    overlap_content = []
                    overlap_tok_count = 0
                    for l in reversed(current_chunk):
                        l_tok = num_tokens(l)
                        if overlap_tok_count + l_tok + 1 <= overlap:
                            overlap_content.insert(0, l)
                            overlap_tok_count += l_tok + 1
                        else:
                            break
                    current_chunk = overlap_content
                    current_tokens = sum(num_tokens(l) + 1 for l in current_chunk) - 1 if current_chunk else 0

                # If a single line is still larger than chunk_size, split by sentences
                if line_tokens > chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', line)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        sentence_tokens = num_tokens(sentence)

                        if current_tokens + sentence_tokens > chunk_size and current_chunk:
                            chunks.append(" ".join(current_chunk))
                            overlap_content = []
                            overlap_tok_count = 0
                            for s in reversed(current_chunk):
                                s_tok = num_tokens(s)
                                if overlap_tok_count + s_tok + 1 <= overlap:
                                    overlap_content.insert(0, s)
                                    overlap_tok_count += s_tok + 1
                                else:
                                    break
                            current_chunk = overlap_content
                            current_tokens = sum(num_tokens(s) + 1 for s in current_chunk) - 1 if current_chunk else 0

                        # If a sentence is still too large, slice it by tokens directly
                        if sentence_tokens > chunk_size:
                            tokens = encoding.encode(sentence)
                            start = 0
                            while start < len(tokens):
                                end = start + chunk_size
                                sub_chunk_tokens = tokens[start:end]
                                sub_chunk = encoding.decode(sub_chunk_tokens).strip()
                                if sub_chunk:
                                    chunks.append(sub_chunk)
                                start += chunk_size - overlap
                            current_chunk = []
                            current_tokens = 0
                        else:
                            current_chunk.append(sentence)
                            current_tokens += sentence_tokens + 2
                else:
                    current_chunk.append(line)
                    current_tokens += line_tokens + 1
        else:
            current_chunk.append(para)
            current_tokens += para_tokens + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


if __name__ == "__main__":
    from utils.pdf import read_pdf
    try:
        text = read_pdf("books/brain_tumour.pdf")
        chunks = chunk_text(text, chunk_size=256, overlap=64)
        print("Chunks:", len(chunks))
        print("Token count of first chunk:", num_tokens(chunks[0]))
        print()
        print("Chunk 0 sample:")
        print(chunks[0][:150])
    except Exception as e:
        print(f"Error test chunking: {e}")