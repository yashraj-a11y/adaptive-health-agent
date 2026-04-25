"""
Knowledge Base Loader

Loads all JSON knowledge base documents from the documents/ directory into
the medical_knowledge_base ChromaDB collection. Checks if documents are
already loaded (via collection.count()) to avoid duplicate insertion.
"""

import os
import json
import glob
from memory.episodic_memory import (
    get_knowledge_base_collection,
    get_embedding
)


def load_knowledge_base():
    """Load all knowledge base JSON files into ChromaDB.

    Reads all .json files from knowledge_base/documents/, generates
    embeddings for each document's searchable content, and inserts
    them into the medical_knowledge_base collection.

    Skips loading if documents already exist in the collection.
    """
    collection = get_knowledge_base_collection()

    # Skip if already loaded
    if collection.count() > 0:
        print(f"[KB Loader] Knowledge base already loaded ({collection.count()} documents). Skipping.")
        return

    # Find all JSON files in the documents directory
    documents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
    json_files = glob.glob(os.path.join(documents_dir, "*.json"))

    if not json_files:
        print(f"[KB Loader] No JSON files found in {documents_dir}")
        return

    all_ids = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []

    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        print(f"[KB Loader] Loading {filename}...")

        with open(json_file, "r") as f:
            entries = json.load(f)

        for entry in entries:
            doc_id = entry["id"]

            # Build searchable text from key fields for embedding generation
            searchable_text = (
                f"{entry['title']}. "
                f"Domain: {entry['domain']}. "
                f"Signals: {', '.join(entry['signals_involved'])}. "
                f"Context: {entry['user_context']}. "
                f"Duration: {entry['duration_context']}. "
                f"{entry['interpretation']}"
            )

            # Build metadata for filtering (ChromaDB metadata must be str/int/float/bool)
            metadata = {
                "domain": entry["domain"],
                "duration_context": entry["duration_context"],
                "user_context": entry["user_context"],
                "title": entry["title"],
                "severity_suggestion": entry["severity_suggestion"],
                "signals": ", ".join(entry["signals_involved"]),
                "recommended_action": entry["recommended_agent_action"],
                "what_not_to_conclude": entry["what_not_to_conclude"],
                "sources": ", ".join(entry["sources"]),
                "source_file": filename
            }

            embedding = get_embedding(searchable_text)

            all_ids.append(doc_id)
            all_embeddings.append(embedding)
            all_documents.append(searchable_text)
            all_metadatas.append(metadata)

    # Batch insert all documents into ChromaDB
    if all_ids:
        collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas
        )
        print(f"[KB Loader] Successfully loaded {len(all_ids)} documents into medical_knowledge_base.")
    else:
        print("[KB Loader] No documents to load.")


def query_knowledge_base(query_text: str, n_results: int = 3) -> list:
    """Query the medical knowledge base for relevant documents.

    Args:
        query_text: Natural language query describing the pattern or situation.
        n_results: Number of results to return.

    Returns:
        list: List of dicts with keys 'id', 'document', 'metadata', 'distance'.
    """
    collection = get_knowledge_base_collection()

    if collection.count() == 0:
        print("[KB Loader] Warning: Knowledge base is empty. Run load_knowledge_base() first.")
        return []

    query_embedding = get_embedding(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count())
    )

    # Format results into a more usable structure
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None
        })

    return formatted
