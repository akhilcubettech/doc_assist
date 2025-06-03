import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from PIL import Image
import pytesseract
import os
import tempfile
from dotenv import load_dotenv
import re


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
    max_tokens=512
)

st.set_page_config(page_title="Doctor's Assistant", layout="wide")
st.title("🏥 Doc Assist: Chat with Your Documents")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# Function to process and embed documents
def process_documents(files):
    documents = []
    for file in files:
        # Skip already processed files
        if file.name in st.session_state.processed_files:
            continue

        if file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.name
            documents.extend(docs)
            os.remove(tmp_file_path)
        elif file.type in ["image/png", "image/jpeg"]:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
            documents.append({
                "page_content": text,
                "metadata": {"source": file.name}
            })

        # Add to processed files
        st.session_state.processed_files.add(file.name)

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        separators=["\n\n", "\n", ".","?", "!", " ", ""]
    )

    if all(isinstance(doc, dict) for doc in documents):
        texts = [doc["page_content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        split_docs = text_splitter.create_documents(texts, metadatas=metadatas)
    else:
        split_docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store


# File uploader in sidebar
with st.sidebar:
    st.subheader("Document Management")
    uploaded_files = st.file_uploader(
        "Upload reference documents (PDF, PNG, JPEG)",
        type=["pdf", "png", "jpeg"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Reset vector store when processing new documents
                new_vector_store = process_documents(uploaded_files)
                if new_vector_store:
                    st.session_state.vector_store = new_vector_store
                    st.success("Documents processed successfully!")
                    # Reset chat history when new documents are processed
                    st.session_state.chat_history = []
                    st.session_state.memory.clear()
                else:
                    st.warning("No new documents to process")
        else:
            st.warning("Please upload documents first")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()

# Display chat history in main area
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.caption(f"Sources: {message['sources']}")


# Query processing with context enhancement
def expand_query(query, chat_history):
    if not chat_history:
        return query

    history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}"
         for msg in st.session_state.chat_history[-4:]]
    )

    prompt = f"""
    Given the conversation history and the current query, expand the query to 
    include relevant context for document retrieval. Focus on medical terms, 
    acronyms, and contextual relationships.

    History:
    {history_str}

    Current Query: {query}

    Expanded Query:
    """

    expanded = llm.invoke(prompt).content
    return expanded.strip()


# Chat interface
if st.session_state.vector_store:
    query = st.chat_input("Ask about your documents:")

    if query:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
            "sources": ""
        })

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(query)

        # Expand query using chat history
        with st.spinner("Analyzing context..."):
            expanded_query = expand_query(query, st.session_state.chat_history)

        # Create conversation chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 15}
            ),
            memory=st.session_state.memory,
            return_source_documents=True
        )

        # Process query
        with st.spinner("Searching documents..."):
            result = qa_chain({"question": expanded_query})

        # Extract sources
        sources = set()
        for doc in result.get("source_documents", []):
            if hasattr(doc, 'metadata') and "source" in doc.metadata:
                source = doc.metadata["source"]
                # Clean up temporary file paths
                if "tmp" in source:
                    source = re.search(r'_(.*?\.pdf)', source).group(1) if re.search(r'_(.*?\.pdf)',
                                                                                     source) else "Uploaded file"
                sources.add(source)

        source_str = ", ".join(sources) if sources else "General knowledge"
        answer = result.get("answer", "I couldn't find an answer in the documents.")

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": source_str
        })

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            if source_str != "General knowledge":
                st.caption(f"Sources: {source_str}")

# Sidebar status
with st.sidebar:
    st.divider()
    st.subheader("Current Status")

    if st.session_state.vector_store:
        st.success("Vector store ready")
        st.write(f"Processed documents: {len(st.session_state.processed_files)}")
    else:
        st.warning("No documents processed")