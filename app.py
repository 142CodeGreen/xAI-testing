import os
import gradio as gr
import logging
import requests
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from requests.exceptions import RequestException
from typing import List

# Configure xAI API settings
XAI_API_KEY = os.getenv('XAI_API_KEY')

class XAIService:
    def __init__(self, base_url: str = "https://api.x.ai/v1", api_key: str = None):
        if api_key is None:
            self.api_key = os.getenv('XAI_API_KEY')
            if not self.api_key:
                raise ValueError("XAI_API_KEY environment variable is not set or api_key not provided")
        else:
            self.api_key = api_key
        self.base_url = base_url

    def get_embedding(self, text: str) -> List[float]:
        endpoint = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        response = requests.post(endpoint, headers=headers, json={"input": text})
        response.raise_for_status()
        return response.json()['data'][0]['embedding']

    def get_response(self, user_message: str, system_message: str = "You are an AI with access to external documents.", model: str = "grok-beta", temperature: float = 0.7) -> str:
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "model": model,
            "temperature": temperature,
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

# Configure settings
xai_service = XAIService(XAI_API_KEY)
Settings.llm = xai_service
Settings.embed_model = lambda text: xai_service.get_embedding(text)  # This now directly uses XAIService's embedding method
Settings.text_splitter = SentenceSplitter(chunk_size=400)

index = None
query_engine = None

def load_documents(file_objs):
    global index, query_engine
    try:
        if not file_objs:
            return "Error: No files selected."

        file_paths = [file_obj.name for file_obj in file_objs]
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()

        if not documents:
            return f"No documents found in the selected files."

        #vector_store = MilvusVectorStore(
        #    host="127.0.0.1",
        #    port=19530,
        #    dim=1024,  # Ensure this matches xAI's embedding model output dimension
        #    collection_name="your_collection_name",
        #    gpu_id=0
        #)
        
        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True,output_fields=[])  # use CPU
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        #index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            embed_model=lambda text: xai_service.get_embedding(text)
        )

        # Create the query engine after the index is created
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

def chat(message, history):
    global query_engine
    if query_engine is None:
        return history + [("Please upload a file first.", None)]

    try:
        # Retrieve relevant documents or context
        response = query_engine.query(message)
        context = "\n".join([node.get_text() for node in response.source_nodes])

        # Use the retrieved context in the user message
        full_prompt = f"Context: {context}\n\nQuestion: {message}"
        
        # Call xAI API for response generation
        ai_response = xai_service.get_response(full_prompt)
        return history + [(message, ai_response)]
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return history + [(message, f"Error processing query: {str(e)}")]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Load Documents")

    load_output = gr.Textbox(label="Load Status")

    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox(label="Enter your question")
    clear = gr.Button("Clear")

    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(chat, [msg, chatbot], chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True, ssl_verify=False)
