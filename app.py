import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
from PyPDF2 import PdfReader

from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from operator import itemgetter


def get_pdf_text(pdf_docs):
    documents = []

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()

            if page_text and page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": pdf.name,
                            "page": page_number
                        }
                    )
                )

    return documents


def get_text_chunks(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    chunks = []

    for document in documents:

        page_chunks = splitter.create_documents(
            [document.page_content],
            metadatas=[document.metadata]
        )

        chunks.extend(
            chunk for chunk in page_chunks
            if chunk.page_content.strip()
        )

    return chunks


def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )


def get_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY was not found in the .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def get_prompt():

    return ChatPromptTemplate.from_template("""
You are an expert cybersecurity tutor helping a student understand
information from uploaded PDF documents.

Answer the user's question using ONLY the retrieved context.

1. Use only information supported by the retrieved context.
2. Do not use outside knowledge.
3. Do not hallucinate or invent facts.
4. Read all retrieved chunks before answering.
5. Combine relevant information from multiple chunks.
6. Explain concepts in simple language.
7. Use numbered points for main ideas.
8. Use bullet points for supporting details.
9. Keep the answer focused on the question.

If the retrieved context does not contain enough information,
respond exactly:

"I couldn't find that information in the uploaded PDFs."

## Answer

Give a direct and concise answer.

## Explanation

1. Explain the first important point.
2. Explain the next important point.
3. Add additional points only when relevant.

## Key Points

- Important point 1
- Important point 2
- Important point 3

## Example

- Provide an example only if it exists in the retrieved context.
- Otherwise write:
  "No example was provided in the uploaded PDFs."

Retrieved Context:
{context}

User Question:
{input}
""")


def get_chain(vectorstore):

    llm = get_llm()
    prompt = get_prompt()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    chain = (
        RunnableParallel(
            context=itemgetter("input") | retriever | format_docs,
            input=itemgetter("input")
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def handle_userinput(question):

    response = st.session_state.chain.invoke(
        {"input": question}
    )

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    return response