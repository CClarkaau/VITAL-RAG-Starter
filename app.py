"""Working baseline chatbot. TODO: add RAG toggle and retrieved context."""
import streamlit as st
from llm import ask_llm
from rag import retrieve, build_context  # ADDED: retrieve policies and format model context.

st.title("Project VITAL — LLM + RAG Lab")
enable_rag = st.sidebar.toggle("Enable RAG", value=False)  # ADDED: switch between baseline and RAG.
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
        # REMOVED: answer = ask_llm(question)
        if enable_rag:  # ADDED
            results = retrieve(question)  # ADDED: find the most relevant policies.
            with st.expander("Retrieved sources", expanded=True):  # ADDED
                if results:  # ADDED
                    for filename, _, _ in results:  # ADDED
                        st.write(filename)  # ADDED: display each retrieved filename.
                else:  # ADDED
                    st.write("No matching policy documents found.")  # ADDED
            context = build_context(results)  # ADDED: empty string preserves RAG rules when no matches.
            answer = ask_llm(question, context=context)  # ADDED: answer using source context.
        else:  # ADDED
            answer = ask_llm(question)  # ADDED: keep the baseline behavior when RAG is disabled.
    except Exception as exc:
        answer = f"Ollama unavailable: {exc}"
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
