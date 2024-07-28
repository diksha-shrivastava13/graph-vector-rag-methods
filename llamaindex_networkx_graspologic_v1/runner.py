import os
import re
from typing import Any

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PropertyGraphIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from graph_rag_store import GraphRAGStore
from graph_rag_extractor import GraphRAGExtractor
from graph_rag_query_engine import GraphRAGQueryEngine

nest_asyncio.apply()

# Load Data (bad, bad idea)
news = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv")[:50]
documents = [Document(text=f"{row['title']}: {row['text']}") for index, row in news.iterrows()]

# Setup API Key and LLM
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
az_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
az_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Models
embed_model = OpenAIEmbedding(embed_batch_size=10)
llm = AzureOpenAI(engine="gpt-4o", model="gpt-4o", temperature=0.0,
                  api_key=az_openai_api_key, azure_endpoint=az_openai_endpoint)

# Configurations
Settings.embed_model = embed_model
Settings.llm = llm


# Step 1: Create nodes/chunks from the text

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)


# Step 2: Build ProperGraphIndex using GraphRAGExtractor and GraphRAGStore

KG_TRIPLET_EXTRACT_TMPL = """
-Goal- Given a text document, identify all entities and their entity types from the text 
and all relationships among the identified entities. Given the text, extract up to {max_knowledge_triplets} 
entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly 
related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each 
other

Format each relationship as ("relationship")

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:
"""

# Regular expression patterns
entity_pattern = r'- entity_name: ([\w\s]+)\n   - entity_type: ([\w\s]+)\n   - entity_description: ((?:[\w\s]+(?:\n(?!\- entity_name).*)?)*)'
relationship_pattern = r'- source_entity: ([\w\s]+)\n   - target_entity: ([\w\s]+)\n   - relation: ([\w\s]+)\n   - relationship_description: ((?:[\w\s]+(?:\n(?!\- source_entity).*)?)*)'


def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str, re.MULTILINE)
    entities = [(name.strip(), type_.strip(), desc.strip()) for name, type_, desc in entities]
    relationships = re.findall(relationship_pattern, response_str, re.MULTILINE)
    relationships = [(source.strip(), target.strip(), relation.strip(), desc.strip()) for source, target, relation, desc
                     in relationships]
    return entities, relationships


kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=2,
    parse_fn=parse_fn,
)
index = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=GraphRAGStore(),
    kg_extractors=[kg_extractor],
    show_progress=True,
)
# print(list(index.property_graph_store.graph.nodes.values())[-1])

index.property_graph_store.build_communities()
query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store, llm=llm
)

response = query_engine.query(
    "What are the main news discussed in the document?"
)
print(response)
