import os
import time
import langchain.text_splitter
import streamlit as st
import pickle
import langchain
import langchain_community
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv() # Pour charger les variables contenues dans .env

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f'Paste below your URL #{i+1}')
    urls.append(url)


process_url_clicked = st.sidebar.button('Process URLs')

main_placefolder = st.empty()

llm = OpenAI(temperature = 0.9, max_tokens = 500)

#Path to save vector store
file_path = "faiss_store_openai"

if process_url_clicked:

    #Load those data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading... Started... ✅ ✅ ✅ ")
    data = loader.load()

    #Split the data to create chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size = 1000,
        chunk_overlap = 100
    )

    main_placefolder.text("Chunk Creation... Started... ✅ ✅ ✅ ")

    docs = text_splitter.split_documents(data)

    #Creating embeddings
    embeddings = OpenAIEmbeddings()
    vectorestore_openai = FAISS.from_documents(docs, embeddings)

    main_placefolder.text("Embedding Vector Started Building... ✅ ✅ ✅ ")
    time.sleep(2)

    
    # Save the FAISS index to a file
    file_path = "faiss_store_openai"
    vectorestore_openai.save_local(file_path)


query = main_placefolder.text_input("Please ask your question:")

if query:
    #Load the vectorstore
    news_vector_store = FAISS.load_local(
    file_path, embeddings, allow_dangerous_deserialization=True
    )

    # Create my chain
    chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = news_vector_store.as_retriever())

    #Prompt my llm chain
    result = chain({"question": query}, return_only_outputs = True)

    #result looks like: {"answer":"", "sources":[]}
    st.header("Answer")
    st.write(result["answer"])

    #Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Here are the references or sources:")
        sources_list = sources.split("\n") #To split the sources by newline if there are many

        for source in sources_list:
            st.write(source)




