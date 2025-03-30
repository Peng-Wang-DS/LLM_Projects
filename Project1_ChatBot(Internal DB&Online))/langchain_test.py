import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import google.generativeai as genai

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
vectordb_file_path = "faiss_index"

# Make GeminiEmbeddings properly inherit from langchain's Embeddings base class
class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name="models/text-embedding-004"):
        self.model_name = model_name
    
    def embed_documents(self, texts):
        # Process batch of documents
        result = []
        for text in texts:
            embedding = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            result.append(embedding["embedding"])
        return result
    
    def embed_query(self, text):
        # Process single query
        embedding = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return embedding["embedding"]
    
    # Add the __call__ method that FAISS is trying to use
    def __call__(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        else:
            return self.embed_query(text)

def create_vector_db():
    """Loads CSV, extracts 'prompt' and 'response', and stores in FAISS."""
    # Load CSV manually with Pandas
    df = pd.read_csv("internal_knowledge_base.csv")

    # Ensure both columns exist
    if "prompt" not in df.columns or "response" not in df.columns:
        raise ValueError("CSV must contain 'prompt' and 'response' columns!")

    # Remove empty rows
    df = df.dropna().reset_index(drop=True)

    # Convert rows into LangChain Document format
    documents = []
    for _, row in df.iterrows():
        prompt_text = row["prompt"].strip()
        response_text = row["response"]

        if not prompt_text or not response_text:
            print(f"Skipping empty row: {row}")
            continue

        documents.append(Document(
            page_content=prompt_text,
            metadata={"response": response_text}
        ))

    if not documents:
        raise ValueError("No valid data found in CSV!")

    # Create embeddings model
    embedding_model = GeminiEmbeddings()
    
    # Create FAISS vector database using the embeddings model
    vectordb = FAISS.from_documents(documents=documents, embedding=embedding_model)
    vectordb.save_local(vectordb_file_path)

    print("FAISS vector database successfully created!")

def get_qa_chain():
    # Create embedding model instance - same class as when creating the DB
    embedding_model = GeminiEmbeddings()
    
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(
        folder_path=vectordb_file_path, 
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    
    # Initialize the Gemini model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Define function to run the QA process
    def run_qa_chain(query):
        try:
            # Use similarity_search directly
            docs = vectordb.similarity_search(query, k=4)
            
            if not docs:
                return {"result": "I don't know.", "source_documents": []}
            
            # Extract context from retrieved documents
            context = "\n\n".join([f"Document {i+1}:\nContent: {doc.page_content}\nResponse: {doc.metadata.get('response', 'No response available')}" 
                                  for i, doc in enumerate(docs)])
            
            # Build the prompt
            prompt = f"""Given the following context and a question, generate an answer based on this context only.
            In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
            If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

            CONTEXT: {context}

            QUESTION: {query}"""
            
            # Generate response using Gemini
            response = model.generate_content(prompt)
            
            return {"result": response.text, "source_documents": docs}
        except Exception as e:
            # Add detailed error handling
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details: {error_details}")
            return {"result": f"Error processing query: {str(e)}", "source_documents": []}
    
    return run_qa_chain

if __name__ == "__main__":
    # create_vector_db()
    qa_function = get_qa_chain()
    result = qa_function("what's the sales value from Pepsi last week?")
    print(result["result"])
    print("\nSource documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"Document {i+1}: {doc.page_content[:100]}...")