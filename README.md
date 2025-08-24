# RAG Pipeline – Local Document Q&A

A side project buid on the skills I gained during my first internship.  
This is a Retrieval Augmented Generation (RAG) based application that lets you process personal or business documents locally with LLMs ensuring privacy and confidentiality.


## ⚙️ Tech Stack
  🔗 LangChain – document loading & chunking  
  🗃️ FAISS – vector store  
  🦙 Ollama (LLaMA 3.2) – LLM  
  🧩 nomic-embed-text – embeddings  
  🎈 Streamlit – frontend  


## Features
- Local & privacy-preserving  
- Source citations for transparency  
- Works on small-scale documents    


## Example Use Case
Like Twitter/X’s **“See similar posts”** — RAG retrieves related chunks and gives grounded answers.


## More Details
👉 Check out my LinkedIn post here: [Your LinkedIn Post Link]


## Run Locally
pip install -r requirements.txt
streamlit run app.py
