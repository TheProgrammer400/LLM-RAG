import chromadb
import concurrent.futures
from pathlib import Path
from ollama import embed, chat

from utils.pdf import read_pdf
from utils.chunking import chunk_text

# ===========================
# Configuration
# ===========================

BOOKS_DIR = Path("books")
DB_DIR = "database"

CHUNK_SIZE = 256  # tokens
OVERLAP = 64     # tokens

# ===========================
# ChromaDB
# ===========================

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("medical_knowledge")


# ===========================
# Helper Functions
# ===========================

def classify_book_category(pdf_text):
    # Sample from the beginning (e.g. title/preface) and the middle (contents) of the PDF
    start_snippet = pdf_text[:2000].strip()
    mid_point = len(pdf_text) // 2
    mid_snippet = pdf_text[mid_point:mid_point + 1500].strip() if len(pdf_text) > 4000 else ""
    snippet = f"{start_snippet}\n\n[...]\n\n{mid_snippet}"


    prompt = (
        "You are a medical literature categorizer.\n\n"
        "Based on the following book title or text snippet, classify the book into exactly ONE of these categories: "
        "neurology, cardiology, pediatrics, oncology, emergency, general.\n\n"
        "Respond with ONLY the single-word category name in lowercase (no punctuation, no explanation, no other words).\n\n"
        f"Text Snippet:\n{snippet}\n\n"
        "Category:"
    )

    try:
        resp = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        category = resp["message"]["content"].strip().lower()
        # Clean any accidental characters
        category = "".join(c for c in category if c.isalnum())

        valid_categories = {"neurology", "cardiology", "pediatrics", "oncology", "emergency", "general"}
        if category in valid_categories:
            return category
        else:
            # Fallback check
            for valid in valid_categories:
                if valid in category:
                    return valid
            return "general"
    except Exception as e:
        print(f"[Warning] LLM auto-categorization failed: {e}")
        return "general"


def process_pdf(pdf_file):
    print(f"[{pdf_file.name}] Reading PDF...")
    try:
        text = read_pdf(pdf_file)
    except Exception as e:
        print(f"[{pdf_file.name}] Failed to read PDF: {e}")
        return None

    print(f"[{pdf_file.name}] Classifying medical category using local LLM...")
    category = classify_book_category(text)
    print(f"[{pdf_file.name}] Auto-categorized as: '{category}'")

    print(f"[{pdf_file.name}] Chunking text into tokens...")
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"[{pdf_file.name}] Generated {len(chunks)} token chunks.")

    print(f"[{pdf_file.name}] Generating embeddings in batches of 32...")
    embeddings = []
    batch_size = 32
    failed = False

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        try:
            batch_resp = embed(
                model="nomic-embed-text",
                input=batch_chunks
            )
            embeddings.extend(batch_resp["embeddings"])
        except Exception as e:
            print(f"[{pdf_file.name}] Ollama embedding API failed for batch: {e}")
            failed = True
            break

    if failed:
        return None

    metadatas = [
        {
            "source": pdf_file.name,
            "category": category
        }
        for _ in chunks
    ]

    return {
        "filename": pdf_file.name,
        "chunks": chunks,
        "embeddings": embeddings,
        "metadatas": metadatas
    }


# ===========================
# Ingestion
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest medical reference PDFs into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Force re-ingest all PDFs even if already in database.")
    parser.add_argument("--clean", action="store_true", help="Clean/reset the database collection before starting.")
    args = parser.parse_args()

    if args.clean:
        print("Cleaning/resetting ChromaDB collection...")
        try:
            client.delete_collection("medical_knowledge")
        except Exception as e:
            print(f"[Warning] Failed to delete collection: {e}")
        # Recreate collection
        collection = client.get_or_create_collection("medical_knowledge")

    doc_id = 0

    # Get the max doc_id from existing collection to ensure unique ids
    existing_ids = collection.get(include=[])["ids"]
    if existing_ids:
        doc_id = max(int(i) for i in existing_ids) + 1

    pdf_files = sorted(BOOKS_DIR.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDFs in books/ directory.\n")

    # Filter out files that are already ingested
    pdf_files_to_process = []
    for pdf_file in pdf_files:
        if args.force:
            pdf_files_to_process.append(pdf_file)
        else:
            existing = collection.get(
                where={"source": pdf_file.name},
                limit=1
            )
            if existing and len(existing["ids"]) > 0:
                print(f"Skipping {pdf_file.name} - already fully ingested.")
            else:
                pdf_files_to_process.append(pdf_file)

    if not pdf_files_to_process:
        print("\nNo new PDFs to process.")
    else:
        print(f"\nProcessing {len(pdf_files_to_process)} PDFs in parallel (max workers = 4)...")
        processed_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_pdf = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files_to_process}

            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    result = future.result()
                    if result:
                        processed_results.append(result)
                except Exception as exc:
                    print(f"[{pdf.name}] generated an exception: {exc}")

        # Sequentially insert items to avoid ChromaDB write locks
        for item in processed_results:
            print(f"Saving chunks from {item['filename']} to ChromaDB...")
            chunk_count = len(item["chunks"])
            ids = [str(doc_id + idx) for idx in range(chunk_count)]
            try:
                collection.add(
                    ids=ids,
                    documents=item["chunks"],
                    embeddings=item["embeddings"],
                    metadatas=item["metadatas"]
                )
                doc_id += chunk_count
                print(f"Finished saving {item['filename']}.")
            except Exception as e:
                print(f"Failed to save {item['filename']} to ChromaDB: {e}")

    print("=" * 60)
    print("Database ingestion process completed!")
    print(f"Current total database chunks: {collection.count()}")