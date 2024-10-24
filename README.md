# PDF Query Using LangChain

## Overview
This project leverages **LangChain** and **HuggingFace Embeddings** to create a Document Question-Answering (Q&A) system for research papers stored in PDF format. It uses **retrieval-augmented generation (RAG)** to answer queries based on the content from the PDF documents. The primary LLM (Language Model) used in this implementation is **ChatGroq**.

## Features
- **Document Embedding**: Uses HuggingFace embeddings to create vector representations of PDF documents for efficient retrieval.
- **Retrieval-Augmented Generation (RAG)**: A combination of information retrieval and language generation techniques to provide accurate answers based on PDF content.
- **Customizable**: Allows users to adjust various parameters such as temperature and max tokens for LLM responses.

## Dependencies
Ensure you have the following Python packages installed:
- `streamlit`
- `langchain-core`
- `langchain-groq`
- `langchain-huggingface`
- `faiss`
- `dotenv`

pip install streamlit langchain-core langchain-groq langchain-huggingface faiss dotenv


## Usage Instructions
## 1. Load and Preprocess Documents
   
Place the research papers you want to query in the research_papers directory. The documents will be loaded and split into chunks for better processing.

3. Start the Streamlit Application
   
To run the application, use the command:

streamlit run PDFQuery_LangChain.ipynb

3. Create Vector Embeddings
   
Click the "Document Embedding" button to process and create vector embeddings of the documents. This will enable accurate retrieval for your queries.

5. Query the Documents
   
Enter your query in the provided text input box. The model will generate a response based on the contents of the PDF documents.

## Code Structure

## Main Components

Document Embedding: Uses HuggingFace's all-MiniLM-L6-v2 model to create vector embeddings of PDF content.
RAG Pipeline: Integrates ChatGroq with a retrieval mechanism using the FAISS vector database for document retrieval.
User Interface: Built with Streamlit to provide an interactive experience for users to query documents.
Key Code Snippets
Loading Documents
python

from langchain_community.document_loaders import PyPDFDirectoryLoader

st.session_state.loader = PyPDFDirectoryLoader("research_papers")
st.session_state.docs = st.session_state.loader.load()
Creating Vector Embeddings
python

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
Generating a Response
python

response = retrieval_chain.invoke({'input': user_prompt})
st.write(response['answer'])


PDFQuery_LangChain/

├── research_papers/          # Directory to store PDF documents

├── PDFQuery_LangChain.ipynb  # Main Jupyter Notebook file

├── .env                      # Environment file to store API keys

└── README.md                 # This markdown file

Environment Variables

## Make sure to create a .env file with the following variables:


GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
