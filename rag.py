"""Student exercise: implement retrieval over the fictional clinic policies."""
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
DOCS = Path(__file__).parent / "knowledge_base"
WORD_TOKENS = TfidfVectorizer().build_tokenizer()
STOP_WORDS = (ENGLISH_STOP_WORDS - {"bill"}) | {"clinic", "fictional", "harbor", "policy"}
WORD_FORMS = {
    "appointments": "appointment",
    "bills": "bill",
    "canceled": "cancel",
    "cancelled": "cancel",
    "canceling": "cancel",
    "cancelling": "cancel",
    "cancellation": "cancel",
    "cancellations": "cancel",
    "communicated": "communicate",
    "results": "result",
}


def tokenize(text: str) -> list[str]:
    """Remove common words and normalize inflections used in clinic policies."""
    return [
        WORD_FORMS.get(word, word)
        for word in WORD_TOKENS(text.lower())
        if word not in STOP_WORDS
    ]

def load_documents(directory=DOCS):
    """Return list of (filename, text) for *.txt files."""
    # TODO 1: Load documents in deterministic filename order.
    return [(path.name, path.read_text(encoding="utf-8"))
            for path in sorted(Path(directory).glob("*.txt")) if path.is_file()]

def retrieve(question: str, k: int = 2, documents=None):
    """Return top-k (filename, text, score) matches using TF-IDF."""
    # TODO 2: Vectorize documents and query, rank cosine similarities.
    # Exclude zero-score matches; break ties by filename.
    if k <= 0 or not question.strip():
        return []
    documents = list(load_documents() if documents is None else documents)
    if not documents:
        return []
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    try:
        document_vectors = vectorizer.fit_transform([text for _, text in documents])
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            return []
        raise
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, document_vectors).ravel()
    hits = [(name, text, float(score))
            for (name, text), score in zip(documents, scores) if score > 0]
    return sorted(hits, key=lambda hit: (-hit[2], hit[0]))[:k]

def build_context(results):
    """Format hits with [Source: filename] headers."""
    # TODO 3: Build the prompt context.
    return "\n\n".join(f"[Source: {name}]\n{text}" for name, text, _ in results)
