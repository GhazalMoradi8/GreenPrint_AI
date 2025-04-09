# Import necessary libraries
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import streamlit as st
import os # Import os for path joining if needed, though not strictly necessary here

# --- Configuration ---

# LLM Configuration
# Ensure your HUGGING_FACE_HUB_TOKEN is set in Streamlit secrets
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(model_name=hf_model, task="text-generation")

# Embeddings Configuration
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
# REMOVED: embeddings_folder = "/content/" # Invalid path for Streamlit Cloud
# Using default cache location for sentence-transformers is recommended for deployment.
embeddings = HuggingFaceEmbedding(model_name=embedding_model)
# REMOVED: cache_folder=embeddings_folder argument above

# Vector Database Configuration
# --- IMPORTANT ---
# 1. Ensure the 'vecctor_index' directory containing your index data
#    is committed and pushed to the ROOT of your GitHub repository.
# 2. Use a relative path to load the index.
persist_directory = "vecctor_index" # Relative path to the index directory in your repo

# Check if persist_directory exists (optional good practice)
if not os.path.exists(persist_directory):
    st.error(f"Error: Vector index directory '{persist_directory}' not found. "
             f"Make sure it's in your GitHub repository root.")
    st.stop() # Stop execution if index is missing

try:
    storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
    vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
except Exception as e:
    st.error(f"Error loading vector index from '{persist_directory}': {e}")
    st.stop()

# Retriever Configuration
retriever = vector_index.as_retriever(similarity_top_k=2)

# Prompt Configuration (System Messages)
prompts = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a nice chatbot having a conversation with a human."),
    ChatMessage(role=MessageRole.SYSTEM, content="Answer the question based only on the following context and previous conversation."),
    ChatMessage(role=MessageRole.SYSTEM, content="Keep your answers short and succinct.")
]

# Memory Configuration
memory = ChatMemoryBuffer.from_defaults()

# --- Bot Initialization ---

# Use Streamlit's caching for the chat engine resource
@st.cache_resource
def init_bot():
    st.write("Initializing Chatbot Engine...") # Add a message to see if this runs
    try:
        engine = ContextChatEngine.from_defaults( # Use from_defaults for easier setup
            llm=llm,
            retriever=retriever,
            memory=memory,
            prefix_messages=prompts,
            verbose=True # Add verbose=True temporarily for debugging if needed
        )
        st.write("Chatbot Engine Initialized.")
        return engine
    except Exception as e:
        st.error(f"Failed to initialize chatbot engine: {e}")
        return None # Return None or handle error appropriately

rag_bot = init_bot()

# Stop if initialization failed
if rag_bot is None:
    st.stop()


# --- Streamlit UI ---

st.title("Carbonfootprint Chatbot")

# Display chat messages from history on app rerun
# Check if rag_bot.chat_history exists and is iterable
if hasattr(rag_bot, 'chat_history') and rag_bot.chat_history:
    for message in rag_bot.chat_history:
        # Check if message has role and content/blocks
        role = getattr(message, 'role', 'unknown')
        content = ""
        if hasattr(message, 'content'):
             content = message.content
        elif hasattr(message, 'blocks') and message.blocks:
             content = getattr(message.blocks[0], 'text', '') # Adapt if block structure differs

        with st.chat_message(role.name if hasattr(role, 'name') else str(role)): # Use role.name if enum
            st.markdown(content)

# React to user input
if prompt := st.chat_input("Curious minds wanted!"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt) # Use "user" role consistent with LlamaIndex

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Digging for answers..."):
        try:
            # Send question to chain to get answer
            answer = rag_bot.chat(prompt)

            # Extract answer from response object
            response_text = getattr(answer, 'response', 'Sorry, I could not process that.')

            # Display chatbot response in chat message container
            with st.chat_message("assistant"): # Use "assistant" role consistent with LlamaIndex
                st.markdown(response_text)
        except Exception as e:
            st.error(f"Error during chat processing: {e}")
            with st.chat_message("assistant"):
                st.markdown("Sorry, an error occurred while trying to get an answer.")
