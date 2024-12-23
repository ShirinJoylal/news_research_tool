import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Check for OpenAI API Key
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API Key not found. Please check your .env file.")
    st.stop()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")  # Button to initiate processing
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()  # Placeholder for main content area
query = st.text_input("Question:")  # Place input outside the block

# Initialize the LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

# Validate URLs and process
if process_url_clicked:
    valid_urls = [url.strip() for url in urls if url.strip()]
    if not valid_urls:
        st.warning("Please enter at least one valid URL.")
        st.stop()

    try:
        loader = UnstructuredURLLoader(urls=valid_urls)
        main_placeholder.text("Data Loading...✅✅✅")
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Split data into smaller documents
    if data:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitting...✅✅✅")
            docs = text_splitter.split_documents(data)

            # Build FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            pkl = vectorstore_openai.serialize_to_bytes()

            # Save to pickle file
            with open(file_path, "wb") as f:
                pickle.dump(pkl, f)
            main_placeholder.text("Embedding Vector Built Successfully...✅✅✅")
        except Exception as e:
            st.error(f"Error during text splitting or vectorization: {str(e)}")
            st.stop()

# Handle Query
if query and os.path.exists(file_path):
    try:
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)
            vectorstore = FAISS.deserialize_from_bytes(embeddings, pkl, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result.get("answer", "No answer found."))

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)
    except Exception as e:
        st.error(f"Error during query processing: {str(e)}")
else:
    st.warning("Please process URLs and enter a question.")
