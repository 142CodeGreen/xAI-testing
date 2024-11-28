import os
import gradio as gr
import logging
import requests
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
#from openai import OpenAI
#import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming xAI provides their endpoint for generating text
XAI_API_URL = "https://api.x.ai/v1/embeddings"

# Custom LLM class for xAI
class XAILLM:
    def __init__(self):
        self.api_key = os.getenv('XAI_API_KEY')
        if not self.api_key:
            raise ValueError("XAI_API_KEY environment variable is not set")

    def complete(self, prompt, **kwargs):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": "grok-beta",  # Adjust model name as per xAI's API
            "messages": [{"role": "user", "content": prompt}],
            "stream": True  # or True if you want streaming responses
        }
        response = requests.post(XAI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']  # Adjust based on actual API response structure

Settings.llm = XAILLM()

# Custom Embedding class for xAI
class XAIEmbedding:
    def __init__(self):
        self.api_key = os.getenv('XAI_API_KEY')
        self.model_id = "v1"  # Adjust as per xAI's model ID

    def get_text_embedding(self, text):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "input": text,
            "model": self.model_id
        }
        response = requests.post(XAI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['data'][0]['embedding']

    def get_text_embedding_batch(self, texts, show_progress=False):
        # If show_progress is set to True, you could implement progress reporting here.
        # However, since the requests library doesn't inherently support this, we'll just print if it's true.
        if show_progress:
            print("Generating embeddings for batch of texts...")
        
        return [self.get_text_embedding(text) for text in texts]
        
Settings.embed_model = XAIEmbedding()
Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Ensure GPU usage
#if torch.cuda.is_available():
#    logger.info("GPU is available and will be used.")
#    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Assuming you want to use GPU 0
#else:
#    logger.warning("GPU not detected or not configured correctly. Falling back to CPU.")

index = None
query_engine = None

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

def load_documents(file_objs):
    global index, query_engine
    try:
        if not file_objs:
            return "Error: No files selected."

        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            directory = os.path.dirname(file_path)
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return f"No documents found in the selected files."

        vector_store = MilvusVectorStore(
            host="127.0.0.1",
            port=19530,
            dim=1024,  # Ensure this matches xAI's embedding model output dimension
            collection_name="your_collection_name",
            gpu_id=0
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create the query engine after the index is created
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

def chat(message, history):
    global query_engine
    if query_engine is None:
        return history + [("Please upload a file first.",None)]
    try:
        response = query_engine.query(message)
        return history + [(message, response.response)]
    except Exception as e:
        return history + [(message,f"Error processing query: {str(e)}")]

def stream_response(message, history):
    global query_engine
    if query_engine is None:
        yield history + [("Please upload a file first.",None)]
        return

    try:
        response = query_engine.query(message)
        for text in response.response_gen:
            yield history + [(message, text)]
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Load Documents")

    load_output = gr.Textbox(label="Load Status")

    chatbot = gr.Chatbot(type='messages')
    #chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question")
    clear = gr.Button("Clear")

    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True, ssl_verify=False)
