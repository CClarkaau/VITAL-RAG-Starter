"""Working baseline chatbot. TODO: add RAG toggle and retrieved context."""
import streamlit as st
from llm import ask_llm
from rag import retrieve, build_context

st.title("Project VITAL — LLM + RAG Lab")
enable_rag = st.sidebar.toggle("Enable RAG", value=False)
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
question = st.chat_input("Ask a question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    # TODO 4: Add an Enable RAG toggle; retrieve documents and pass context.
    try:
        if enable_rag:
            results = retrieve(question)
            with st.expander("Retrieved sources", expanded=True):
                if results:
                    for filename, _, _ in results:
                        st.write(filename)
                else:
                    st.write("No matching policy documents found.")
            context = build_context(results)
            answer = ask_llm(question, context=context)
        else:
            answer = ask_llm(question)
    except Exception as exc:
        answer = f"Ollama unavailable: {exc}"
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
