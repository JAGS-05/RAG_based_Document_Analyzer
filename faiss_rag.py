import stat
import re
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM 

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

DATA_PATH = r"RAG\data"
FAISS_INDEX_PATH = r"RAG\faiss_index" 

def load_documents(filename : dir):
    """Loads PDF documents from the data directory."""
    documents = []
    if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_PATH, filename)
            try:
                loader = PyPDFLoader(filepath)
                loaded_documents = loader.load()
                for i, doc in enumerate(loaded_documents):
                    doc.metadata['source'] = filename 
                    doc.metadata['page'] = i + 1
                documents.extend(loaded_documents)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    return documents

def clean_text(text : str):
    """Cleans the text by removing unwanted spaces or newlines."""
    text = re.sub(r'\n+', '\n', text)  
    text = re.sub(r'\s{2,}', ' ', text)  

    return text.strip()

def split_documents(documents : Document):
    """Cleans and splits documents into smaller chunks."""
    cleaned_docs = []
    for doc in documents:
        cleaned = clean_text(doc.page_content)
        cleaned_docs.append(Document(page_content=cleaned, metadata=doc.metadata))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(cleaned_docs)

def add_to_faiss(chunks: list):
    """Store documents in FAISS vectorstore"""
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        
        print("Creating FAISS index...")

        db = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        db.save_local(FAISS_INDEX_PATH)
        print(f"Saved FAISS index to '{FAISS_INDEX_PATH}'")
        
        return db
        
    except Exception as e:
        print(f"FAISS creation failed: {str(e)}")
        return None

def load_faiss():
    """Load existing FAISS index"""
    try:
        if not os.path.exists(FAISS_INDEX_PATH):
            print("No existing FAISS index found")
            return None
            
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Loaded index with {db.index.ntotal} vectors")
        return db
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        return None

PROMPT_TEMPLATE = """
Answer the question based ONLY on the below context:

{context}

-------

Question: {question}
"""

def query_rag(query_text: str):
    try:
        embedding_function = OllamaEmbeddings(model='nomic-embed-text')
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_function,
            allow_dangerous_deserialization=True
        )

        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if not results:
            return "No relevant results found in documents."

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        print(f'CONTEXT TEXT:\n{context_text}\n')

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(f'PROMPT:\n{prompt}\n')

        model = OllamaLLM(model="llama3.2:3b", temperature=0.3)
        response_text = model.invoke(prompt)

        sources = [doc.metadata.get("page", "unknown") for doc, _score in results]
        formatted_response = f"{response_text}\n\nSources: {sources}"
        
        return formatted_response

    except Exception as e:
        print(f"Error during RAG query: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

def main():
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    db = load_faiss() or add_to_faiss(chunks)
    print(f"Added {len(chunks)} chunks to FAISS index")
    
    if db:
        print(f"Index ready with {db.index.ntotal} vectors")
    else:
        print("Failed to initialize FAISS index")
    print("\nRAG pipeline setup - Complete\n")


if __name__ == "__main__":
    main()
    # while True:
    #     query = input("\nEnter your question (or 'quit' to exit): ")
    #     if query.lower() == 'quit':
    #         break
    #     query_rag(query)