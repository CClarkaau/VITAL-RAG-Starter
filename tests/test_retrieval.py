from rag import load_documents, retrieve, build_context

def test_load_documents():
    assert len(load_documents()) == 6

def test_cancellation_policy_retrieved():
    hits = retrieve("How many hours before an appointment must I cancel?", k=2)
    assert "cancellations.txt" in [name for name, _, _ in hits]

def test_context_includes_source():
    assert "[Source: cancellations.txt]" in build_context([("cancellations.txt", "18 hours", 0.9)])

def test_empty_corpus():
    assert retrieve("question", documents=[]) == []

def test_empty_question():
    assert retrieve("", k=2) == []
    assert retrieve("   ", k=2) == []


def test_free_transportation_has_no_policy_match():
    assert retrieve("Does the clinic provide free transportation?") == []


def test_cancellation_matches_across_word_forms():
    hits = retrieve("When can I cancel?")
    assert hits[0][0] == "cancellations.txt"


def test_retrieve_returns_ranked_results_within_limit():
    questions = [
        "How are laboratory results communicated?",
        "Does the clinic provide free transportation?",
    ]

    for question in questions:
        all_hits = retrieve(question, k=6)
        top_two = retrieve(question, k=2)

        assert len(top_two) <= 2, question
        assert top_two == all_hits[:2], question

        scores = [score for _, _, score in all_hits]
        assert scores == sorted(scores, reverse=True), question
