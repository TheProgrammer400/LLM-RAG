import chromadb
from ollama import embed

client = chromadb.PersistentClient(path="database")

SIMILARITY_THRESHOLD = 0.7


def retrieve(query, category=None, top_k=5):
    try:
        collection = client.get_collection("medical_knowledge")
    except Exception:
        # Collection has not been created/ingested yet
        return {
            "documents": [],
            "metadatas": [],
            "distances": []
        }

    try:
        query_embedding = embed(
            model="nomic-embed-text",
            input=query
        )["embeddings"][0]
    except Exception as e:
        print(f"\n[Warning] Ollama embedding failed: {e}")
        return {
            "documents": [],
            "metadatas": [],
            "distances": []
        }

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if category is not None:
        query_args["where"] = {
            "category": category
        }

    try:
        results = collection.query(**query_args)
    except Exception as e:
        print(f"\n[Warning] ChromaDB query failed: {e}")
        return {
            "documents": [],
            "metadatas": [],
            "distances": []
        }

    filtered_documents = []
    filtered_metadata = []
    filtered_distances = []

    if not results or "documents" not in results or not results["documents"] or not results["documents"][0]:
        return {
            "documents": [],
            "metadatas": [],
            "distances": []
        }

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if distance <= SIMILARITY_THRESHOLD:
            filtered_documents.append(document)
            filtered_metadata.append(metadata)
            filtered_distances.append(distance)

    return {
        "documents": filtered_documents,
        "metadatas": filtered_metadata,
        "distances": filtered_distances
    }