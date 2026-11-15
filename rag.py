"""Student exercise: implement retrieval over the fictional clinic policies."""
from pathlib import Path
DOCS = Path(__file__).parent / "knowledge_base"

def load_documents(directory=DOCS):
    """Return list of (filename, text) for *.txt files."""
    # TODO 1: Load documents in deterministic filename order.
    raise NotImplementedError("Implement load_documents")

def retrieve(question: str, k: int = 2, documents=None):
    """Return top-k (filename, text, score) matches using TF-IDF."""
    # TODO 2: Vectorize documents and query, rank cosine similarities.
    # Exclude zero-score matches; break ties by filename.
    raise NotImplementedError("Implement retrieve")

def build_context(results):
    """Format hits with [Source: filename] headers."""
    # TODO 3: Build the prompt context.
    raise NotImplementedError("Implement build_context")
