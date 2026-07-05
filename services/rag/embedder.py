# Embedder — turns text into vectors. Configurable provider (local default),
# dimension follows config. Third stage of the RAG ingest pipeline; also used
# by the retriever to embed queries.

from __future__ import annotations

from functools import lru_cache

from ops_common.config import settings
from ops_common.logging import get_logger

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Local sentence-transformers backend
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _local_model():
    # Loaded once per process and cached. Model is baked into the image at build
    # time, so this does not hit the network at runtime.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingError(
            "sentence-transformers not installed; cannot use local embeddings"
        ) from exc

    model_name = settings.embedding_model
    logger.info("Loading local embedding model", extra={"model": model_name})
    return SentenceTransformer(model_name)


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _local_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine-ready unit vectors
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# API backends (optional future providers, config-switchable)
# ---------------------------------------------------------------------------

def _embed_openai(texts: list[str]) -> list[list[float]]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingError("openai SDK not installed") from exc
    client = OpenAI()
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def _embed_voyage(texts: list[str]) -> list[list[float]]:
    try:
        import voyageai
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingError("voyageai SDK not installed") from exc
    client = voyageai.Client()
    result = client.embed(texts, model=settings.embedding_model)
    return result.embeddings


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _dispatch(texts: list[str]) -> list[list[float]]:
    provider = settings.embedding_provider.lower()
    if provider == "local":
        return _embed_local(texts)
    if provider == "openai":
        return _embed_openai(texts)
    if provider == "voyage":
        return _embed_voyage(texts)
    raise EmbeddingError(f"Unknown embedding provider: {provider}")


def _validate_dimension(vectors: list[list[float]]) -> None:
    if not vectors:
        return
    dim = len(vectors[0])
    expected = settings.embedding_dimension
    if dim != expected:
        # A mismatch here means the model and the rag.embeddings VECTOR(n) column
        # disagree — inserts would fail. Fail loudly and early instead.
        raise EmbeddingError(
            f"Embedding dimension {dim} != configured {expected}. "
            f"Update OPS_EMBEDDING_DIMENSION and the rag.embeddings column to match."
        )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into vectors. Empty strings are embedded as-is."""
    if not texts:
        return []
    cleaned = [t if t and t.strip() else " " for t in texts]
    try:
        vectors = _dispatch(cleaned)
    except EmbeddingError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Embedding failed")
        raise EmbeddingError(str(exc)) from exc
    _validate_dimension(vectors)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string; returns one vector."""
    vectors = embed_texts([text])
    return vectors[0] if vectors else []


def model_name() -> str:
    return settings.embedding_model