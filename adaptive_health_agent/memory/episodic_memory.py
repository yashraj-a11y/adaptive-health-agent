"""
Episodic Memory Module

ChromaDB-backed storage for health episodes and medical knowledge.
Uses PersistentClient for durable local storage and SentenceTransformer
(all-MiniLM-L6-v2) for embedding generation.

Two collections:
  - "episodic_memory": Stores health episodes (anomalies, patterns, conversations)
  - "medical_knowledge_base": Stores clinical knowledge documents for RAG
"""

import os
import chromadb
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Initialize the embedding model (loaded once at module import)
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB persistent client
_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Create or retrieve collections with cosine similarity
episodic_collection = _chroma_client.get_or_create_collection(
    name="episodic_memory",
    metadata={"hnsw:space": "cosine"}
)

knowledge_base_collection = _chroma_client.get_or_create_collection(
    name="medical_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)


def get_embedding(text: str) -> list:
    """Generate an embedding vector for the given text string.

    Args:
        text: Input text to embed.

    Returns:
        list: Embedding vector as a list of floats.
    """
    return _embedding_model.encode(text).tolist()


def get_chroma_client():
    """Return the ChromaDB PersistentClient instance."""
    return _chroma_client


def get_episodic_collection():
    """Return the episodic_memory ChromaDB collection."""
    return episodic_collection


def get_knowledge_base_collection():
    """Return the medical_knowledge_base ChromaDB collection."""
    return knowledge_base_collection


def log_episode(episode: dict) -> None:
    """Log a health episode to the episodic memory collection.

    Args:
        episode: Dict with the episode structure:
            - id: str (format: "episode_{timestamp}_{user_id}")
            - timestamp: str (ISO format)
            - user_id: str
            - event_type: str ("anomaly"|"pattern"|"conversation"|"weekly_summary"|"onboarding")
            - metrics_snapshot: dict
            - context_snapshot: dict
            - deviation_from_baseline: dict
            - significance: str ("pattern_confirmed"|"single_occurrence"|"resolved"|"ongoing")
            - agent_action_taken: str
            - user_response: str or None
            - outcome: str or None
            - tags: list of str
    """
    # Build a searchable text document from the episode content
    searchable_parts = [
        f"Event type: {episode.get('event_type', 'unknown')}",
        f"Significance: {episode.get('significance', 'unknown')}",
        f"Action taken: {episode.get('agent_action_taken', 'none')}",
    ]

    # Add metrics snapshot info to searchable text
    metrics = episode.get("metrics_snapshot", {})
    if metrics:
        metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
        searchable_parts.append(f"Metrics: {metrics_str}")

    # Add deviation info
    deviations = episode.get("deviation_from_baseline", {})
    if deviations:
        dev_str = ", ".join(f"{k}: {v}" for k, v in deviations.items())
        searchable_parts.append(f"Deviations: {dev_str}")

    # Add tags
    tags = episode.get("tags", [])
    if tags:
        searchable_parts.append(f"Tags: {', '.join(tags)}")

    # Add context
    context = episode.get("context_snapshot", {})
    if context:
        ctx_str = ", ".join(f"{k}: {v}" for k, v in context.items())
        searchable_parts.append(f"Context: {ctx_str}")

    searchable_text = ". ".join(searchable_parts)

    # Build metadata (ChromaDB requires str/int/float/bool values)
    metadata = {
        "timestamp": episode.get("timestamp", ""),
        "user_id": episode.get("user_id", ""),
        "event_type": episode.get("event_type", ""),
        "significance": episode.get("significance", ""),
        "agent_action_taken": episode.get("agent_action_taken", ""),
        "user_response": episode.get("user_response", "") or "",
        "outcome": episode.get("outcome", "") or "",
        "tags": ", ".join(episode.get("tags", [])),
    }

    embedding = get_embedding(searchable_text)

    episodic_collection.add(
        ids=[episode["id"]],
        embeddings=[embedding],
        documents=[searchable_text],
        metadatas=[metadata]
    )


def query_similar(query_text: str, n_results: int = 3, where: dict = None) -> list:
    """Query the episodic memory for similar episodes using semantic search.

    Args:
        query_text: Natural language description to search for.
        n_results: Maximum number of results to return.
        where: Optional ChromaDB where filter dict (e.g., {"user_id": "user_a"}).

    Returns:
        list: List of dicts with keys 'id', 'document', 'metadata', 'distance'.
    """
    if episodic_collection.count() == 0:
        return []

    query_embedding = get_embedding(query_text)

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, episodic_collection.count()),
    }
    if where:
        query_params["where"] = where

    results = episodic_collection.query(**query_params)

    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None
        })

    return formatted


def get_recent(user_id: str, days: int = 7) -> list:
    """Retrieve recent episodes for a user within the last N days.

    Args:
        user_id: The user identifier to filter by.
        days: Number of days to look back (default 7).

    Returns:
        list: List of dicts with episode data, sorted by timestamp descending.
    """
    if episodic_collection.count() == 0:
        return []

    # Query with user_id filter — retrieve a generous number and filter by date
    results = episodic_collection.get(
        where={"user_id": user_id},
        include=["documents", "metadatas"]
    )

    if not results["ids"]:
        return []

    # Filter by timestamp within the requested day range
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    recent = []
    for i in range(len(results["ids"])):
        metadata = results["metadatas"][i]
        ts = metadata.get("timestamp", "")
        if ts >= cutoff_str:
            recent.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": metadata
            })

    # Sort by timestamp descending (most recent first)
    recent.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)

    return recent


def update_episode(episode_id: str, updates: dict) -> None:
    """Update metadata fields on an existing episode.

    Args:
        episode_id: The episode ID to update.
        updates: Dict of metadata key-value pairs to update.
    """
    # Retrieve the existing episode
    existing = episodic_collection.get(
        ids=[episode_id],
        include=["metadatas", "documents", "embeddings"]
    )

    if not existing["ids"]:
        print(f"[Episodic Memory] Episode not found: {episode_id}")
        return

    # Merge updates into existing metadata
    current_metadata = existing["metadatas"][0]
    for key, value in updates.items():
        # ChromaDB metadata values must be str/int/float/bool
        if value is None:
            value = ""
        current_metadata[key] = value

    # Update the episode in ChromaDB
    episodic_collection.update(
        ids=[episode_id],
        metadatas=[current_metadata]
    )
