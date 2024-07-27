import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse

# Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
az_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
az_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# models
embed_model = OpenAIEmbedding(embed_batch_size=10)
llm = AzureOpenAI(engine="gpt-4o", model="gpt-4o", temperature=0.0,
                  api_key=az_openai_api_key, azure_endpoint=az_openai_endpoint)

# Configurations
Settings.embed_model = embed_model
Settings.llm = llm

# Streamlit App
st.set_page_config(
    page_title="Data Aggregation",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
st.markdown("<h1 style='text-align: center;'>📈 Data Aggregation </h1>", unsafe_allow_html=True)
st.markdown(" ")
st.markdown("<p style='text-align: center;'>Upload your Document to Portfolio Navigator and get overview, insights, "
            "history and analysis.", unsafe_allow_html=True)

# Upload functionality
uploaded_files = st.file_uploader("Choose your PDF files", type="pdf", accept_multiple_files=True)
processing_container = st.empty()
processing_container.text(" ")

file_path_array = []
if uploaded_files:
    for index, file in enumerate(uploaded_files):
        st.markdown("Processing:", file.name)
        temp_filepath = Path("./local_storage/" + f"{file.name}")
        file_path_array.append("./local_storage/" + f"{file.name}")
        with open(temp_filepath, "wb") as f:
            f.write(file.read())

    processing_container.markdown("Files uploaded...")
    time.sleep(2)
else:
    st.markdown("""<p style="color: #3ae2a5;">Please upload a PDF file.</p>""", unsafe_allow_html=True)


# data processing
parser = LlamaParse(result_type="markdown", num_workers=8)
documents = parser.load_data(file_path_array)


