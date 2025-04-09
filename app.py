import streamlit as st
import gdown
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from huggingface_hub import login

def set_hf_cache():
    os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.makedirs("/tmp/hf_cache", exist_ok=True)
set_hf_cache()  # Call this early in your app
# os.environ["HF_HOME"] = "/tmp/hf_cache"
# os.makedirs("/tmp/hf_cache", exist_ok=True)
# Now use your HF token
hf_token = st.secrets["Amu_ocha"]
login(token=st.secrets["Amu_ocha"])
login(token=hf_token)

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2",
                filename="config.json",
                token=hf_token,
                local_files_only=False)


# Download the vector database when the app starts
st.title("RAG with Streamlit and Hugging Face")

if st.button('Download Vector Database'):
    gdown.download_folder('https://drive.google.com/drive/folders/184-9oriLLmCtsTGvNSKJ5fp1YxZyDWAp?usp=drive_link', quiet=False)
    st.success("Vector database downloaded successfully!")
# hf_token = st.secrets["Amu_ocha"]
# st.secrets["hf_token"] == "Amu_ocha"
# llm
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceInferenceAPI(model_name = hf_model, task = "text-generation")

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/content/"

embeddings = HuggingFaceEmbedding(model_name=embedding_model,
                                  cache_folder=embeddings_folder)

# load Vector Database
storage_context = StorageContext.from_defaults(persist_dir="/content/vector_index")
vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

# retriever
retriever = vector_index.as_retriever(similarity_top_k=2)

# prompt
prompts = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a nice chatbot having a conversation with a human."),
    ChatMessage(role=MessageRole.SYSTEM, content="Answer the question based only on the following context and previous conversation."),
    ChatMessage(role=MessageRole.SYSTEM, content="Keep your answers short and succinct.")
]

# memory
memory = ChatMemoryBuffer.from_defaults()

# bot with memory
@st.cache_resource
def init_bot():
    return ContextChatEngine(llm=llm, retriever=retriever, memory=memory, prefix_messages=prompts)

rag_bot = init_bot()


##### streamlit #####

st.title("Carbonfootprint Chatbot")


# Display chat messages from history on app rerun
for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)

# React to user input
if prompt := st.chat_input("Curious minds wanted!"):

    # Display user message in chat message container
    st.chat_message("human").markdown(prompt)

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Digging for answers..."):

        # send question to chain to get answer
        answer = rag_bot.chat(prompt)

        # extract answer from dictionary returned by chain
        response = answer.response

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
