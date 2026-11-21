"""Student exercise: implement retrieval over the fictional clinic policies."""
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  # ADDED: compare the question with documents.
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
    # REMOVED: raise NotImplementedError("Implement load_documents")
    return [(path.name, path.read_text(encoding="utf-8"))  # ADDED: load each filename and its text.
            for path in sorted(Path(directory).glob("*.txt")) if path.is_file()]  # ADDED: deterministic file order.

def retrieve(question: str, k: int = 2, documents=None):
    """Return top-k (filename, text, score) matches using TF-IDF."""
    # TODO 2: Vectorize documents and query, rank cosine similarities.
    # Exclude zero-score matches; break ties by filename.
    # REMOVED: raise NotImplementedError("Implement retrieve")
    if k <= 0 or not question.strip():  # ADDED: no results requested or no question.
        return []  # ADDED
    documents = list(load_documents() if documents is None else documents)  # ADDED: allow an explicit corpus.
    if not documents:  # ADDED: handle an empty corpus.
        return []  # ADDED
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    try:  # ADDED: text containing no usable tokens has no matches.
        document_vectors = vectorizer.fit_transform([text for _, text in documents])  # ADDED
    except ValueError as exc:  # ADDED
        if "empty vocabulary" in str(exc):  # ADDED
            return []  # ADDED
        raise  # ADDED: preserve unexpected vectorization errors.
    question_vector = vectorizer.transform([question])  # ADDED: use the document vocabulary.
    scores = cosine_similarity(question_vector, document_vectors).ravel()  # ADDED
    hits = [(name, text, float(score))  # ADDED
            for (name, text), score in zip(documents, scores) if score > 0]  # ADDED: omit zero scores.
    return sorted(hits, key=lambda hit: (-hit[2], hit[0]))[:k]  # ADDED: rank, break ties, limit results.

def build_context(results):
    """Format hits with [Source: filename] headers."""
    # TODO 3: Build the prompt context.
    # REMOVED: raise NotImplementedError("Implement build_context")
    return "\n\n".join(f"[Source: {name}]\n{text}" for name, text, _ in results)  # ADDED: label sources.
