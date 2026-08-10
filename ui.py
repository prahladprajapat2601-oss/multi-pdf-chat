import streamlit as st

from app import (
    get_pdf_text,
    get_text_chunks,
    get_vectorstore,
    get_chain,
    handle_userinput
)


def main():

    st.set_page_config(
        page_title="PDF Study Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # -----------------------------
    # Session State
    # -----------------------------

    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # -----------------------------
    # Sidebar
    # -----------------------------

    with st.sidebar:

        st.title("📚 PDF Study Assistant")

        st.caption(
            "Upload PDFs and ask questions about their content."
        )

        st.divider()

        st.subheader("📄 Upload Documents")

        pdf_docs = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if pdf_docs:

            st.write(
                f"**{len(pdf_docs)} PDF(s) selected**"
            )

            for pdf in pdf_docs:
                st.caption(f"📄 {pdf.name}")

        else:

            st.info(
                "Upload one or more PDFs to get started."
            )

        st.divider()

        # -----------------------------
        # Process PDFs
        # -----------------------------

        if st.button(
            "⚡ Process Documents",
            type="primary",
            use_container_width=True
        ):

            if not pdf_docs:

                st.warning(
                    "Please upload at least one PDF first."
                )

            else:

                try:

                    with st.spinner(
                        "Processing your PDFs..."
                    ):

                        documents = get_pdf_text(
                            pdf_docs
                        )

                        chunks = get_text_chunks(
                            documents
                        )

                        vectorstore = get_vectorstore(
                            chunks
                        )

                        st.session_state.chain = get_chain(
                            vectorstore
                        )

                        st.session_state.processed_files = [
                            pdf.name
                            for pdf in pdf_docs
                        ]

                        st.session_state.chat_history = []

                    st.success(
                        "✅ PDFs processed successfully!"
                    )

                except Exception as e:

                    st.error(
                        "Something went wrong."
                    )

                    st.exception(e)

        # -----------------------------
        # Status
        # -----------------------------

        st.divider()

        if st.session_state.chain:

            st.success("🟢 Documents ready")

            st.subheader("Processed PDFs")

            for file_name in st.session_state.processed_files:
                st.caption(f"✓ {file_name}")

            if st.button(
                "🗑️ Clear Conversation",
                use_container_width=True
            ):

                st.session_state.chat_history = []

                st.rerun()

        else:

            st.warning(
                "No documents processed yet."
            )

    # -----------------------------
    # Main Chat
    # -----------------------------

    st.title("💬 Chat with Your PDFs")

    if not st.session_state.chain:

        st.markdown(
            """
### 👋 Welcome!

Upload your PDF documents from the sidebar.

Then click **Process Documents** and start asking questions.
"""
        )

    else:

        st.caption(
            "Ask questions about your uploaded documents."
        )

    # -----------------------------
    # Chat History
    # -----------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # -----------------------------
    # Chat Input
    # -----------------------------

    question = st.chat_input(
        "Ask something about your PDFs..."
    )

    if question:

        if st.session_state.chain is None:

            st.warning(
                "Please process your PDFs first."
            )

            return

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching your PDFs..."):

                try:

                    response = handle_userinput(
                        question
                    )

                    st.markdown(response)

                except Exception as e:

                    st.error(
                        "Could not generate the answer."
                    )

                    st.exception(e)


if __name__ == "__main__":
    main()