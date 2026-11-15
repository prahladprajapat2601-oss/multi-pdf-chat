import os
import re
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

FALLBACK_RESPONSE = (
    "I couldn't find that information in the uploaded PDFs."
)


def get_pdf_text(pdf_docs):

    documents = []

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page_number, page in enumerate(
            pdf_reader.pages, start=1
        ):

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    return chunks


def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )

    return vectorstore


def get_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:

        raise ValueError(
            "GOOGLE_API_KEY was not found in the .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.2
    )


def get_evaluator_prompt():

    return ChatPromptTemplate.from_template("""
You are a document relevance evaluator in a
Corrective Retrieval-Augmented Generation system.

Your task is to evaluate the retrieved documents
against the user's question.

USER QUESTION:
{question}

RETRIEVED DOCUMENTS:
{documents}

INSTRUCTIONS:

1. Evaluate every document independently.
2. Mark a document RELEVANT if it contains information
   directly useful for answering the question.
3. Otherwise mark it IRRELEVANT.
4. Evaluate whether the collection of relevant documents
   contains sufficient information to answer the question.
5. Do not use outside knowledge.
6. Do not assume that missing information is present.

Return your result in exactly this format:

DOC 1: RELEVANT
DOC 2: IRRELEVANT
DOC 3: RELEVANT

SUFFICIENT: YES

Replace the document numbers with the actual document
numbers provided.

If the retrieved information is insufficient, return:

SUFFICIENT: NO

Do not include any additional explanation.
""")


def evaluate_documents(question, docs, llm):

    if not docs:
        return [], False



    formatted_documents = "\n\n".join(
        f"DOC {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    evaluator_prompt = get_evaluator_prompt()

    evaluator_chain = (
        evaluator_prompt
        | llm
        | StrOutputParser()
    )

    evaluation = evaluator_chain.invoke(
        {
            "question": question,
            "documents": formatted_documents
        }
    )

    

    relevant_docs = []

    for i, doc in enumerate(docs, start=1):

        pattern = (
            rf"DOC\s*{i}\s*:\s*RELEVANT\b"
        )

        if re.search(
            pattern,
            evaluation,
            re.IGNORECASE
        ):

            relevant_docs.append(doc)

   

    sufficient = bool(
        re.search(
            r"SUFFICIENT\s*:\s*YES\b",
            evaluation,
            re.IGNORECASE
        )
    )



    labels_found = re.findall(
        r"DOC\s*\d+\s*:\s*(?:RELEVANT|IRRELEVANT)\b",
        evaluation,
        re.IGNORECASE
    )

    if len(labels_found) != len(docs):
        return [], False

    return relevant_docs, sufficient



def get_prompt():

    return ChatPromptTemplate.from_template("""
You are an expert cybersecurity tutor helping a
student understand information from uploaded PDF documents.

Answer the user's question using ONLY the retrieved context.

RULES:

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

Explain the important points step by step.

## Key Points

- Important point 1
- Important point 2
- Important point 3

## Example

Provide an example only if it exists in the retrieved context.

Otherwise write:

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

   

    generation_chain = (
        prompt
        | llm
        | StrOutputParser()
    )



    def run_crag(inputs):

        question = inputs["input"]


        retrieved_docs = retriever.invoke(question)


        relevant_docs, sufficient = evaluate_documents(
            question,
            retrieved_docs,
            llm
        )

        if not sufficient:



            additional_docs = vectorstore.similarity_search(
                question,
                k=12
            )


            combined_docs = []
            seen_content = set()

            for doc in retrieved_docs + additional_docs:

                content_key = doc.page_content.strip()

                if content_key and content_key not in seen_content:

                    combined_docs.append(doc)
                    seen_content.add(content_key)


            relevant_docs, sufficient = evaluate_documents(
                question,
                combined_docs,
                llm
            )



        if not sufficient or not relevant_docs:

            return FALLBACK_RESPONSE


        context = format_docs(relevant_docs)


        response = generation_chain.invoke(
            {
                "context": context,
                "input": question
            }
        )

        return response

    return RunnableLambda(run_crag)



def handle_userinput(question):

    response = st.session_state.chain.invoke(
        {
            "input": question
        }
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
