import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from htmlTemplates import css, bot_template, user_template


# -------------------------------
# Read PDFs
# -------------------------------

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# -------------------------------
# Split text
# -------------------------------

def get_text_chunks(raw_text):

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

    return splitter.split_text(raw_text)


# -------------------------------
# Vector Store
# -------------------------------

def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


# -------------------------------
# Gemini
# -------------------------------

def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )


# -------------------------------
# Prompt
# -------------------------------

def get_prompt():

    prompt = ChatPromptTemplate.from_template("""
You are an expert cybersecurity tutor.

Answer the user's question ONLY using the retrieved context.

Instructions:
1. Read all the retrieved context before answering.
2. If multiple context chunks contain relevant information, combine them into one complete answer.
3. Do not answer using only a single example if the context contains a broader explanation.
4. Explain the answer in simple language suitable for a student.
5. Use bullet points whenever appropriate.
6. If the context does not contain the answer, reply exactly:
"I couldn't find that information in the uploaded PDFs."

Respond in the following format:

## Answer
Provide a concise answer.

## Explanation
Explain the concept in detail using only the retrieved context.

## Key Points
- Point 1
- Point 2
- Point 3

## Example
Provide an example only if one exists in the retrieved context.

Context:
{context}

Question:
{input}
""")

    return prompt


# -------------------------------
# Build Chain
# -------------------------------

def get_chain(vectorstore):

    llm = get_llm()

    prompt = get_prompt()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


# -------------------------------
# User Input
# -------------------------------

def handle_userinput(question):

    response = st.session_state.chain.invoke(
        {"input": question}
    )

    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response["answer"]}
    )

# -------------------------------
# Main
# -------------------------------

def main():

    load_dotenv()

    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📚"
    )

    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("📚 Chat with Multiple PDFs")

    # Display previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input("Ask a question about your PDFs")

    if question:

        if st.session_state.chain is None:
            st.warning("Please upload PDFs first.")
        else:
            handle_userinput(question)

    # Sidebar
    with st.sidebar:

        st.subheader("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process"):

            if pdf_docs:

                with st.spinner("Processing..."):

                    raw_text = get_pdf_text(pdf_docs)

                    chunks = get_text_chunks(raw_text)

                    vectorstore = get_vectorstore(chunks)

                    st.session_state.chain = get_chain(vectorstore)

                st.success("✅ PDFs processed successfully!")

            else:
                st.warning("Please upload at least one PDF.")

if __name__ == "__main__":
    main()