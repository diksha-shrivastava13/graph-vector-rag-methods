"""
GraphRAG combines the strengths of Retrieval Augmented Generation (RAG) and Query-Focused
Summarization (QFS) to effectively handle complex queries over large text datasets.

* RAG excels in fetching precise information, struggles with broader queries that require
thematic understanding.
* QFS can tackle issues that require thematic understanding but cannot scale well.

GraphRAG integrates these two to tackle thematic problem understanding and can scale well.

Steps to Build GraphRAG:

1. Graph Generation:
    a. Source Documents to Text Chunks              - Implemented with SentenceSplitter
    b. Text Chunks to Element Summaries             - Implemented with GraphRAGExtractor
    c. Element Instances to Element Summaries       - Implemented with GraphRAGExtractor
    d. Element Summaries to Graph Communities       _ implemented using GraphRAGStore
    e. Graph Communities to Community Summaries     - Implemented with GraphRAGStore
2. Answering the Query:
    a. Community Summaries to Global Answers.       - Implemented with GraphQueryEngine
"""

# Imports
import pandas as pd
import os
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.llms.azure_openai import AzureOpenAI

# Imports: GraphRAGExtractor
import asyncio
import nest_asyncio

from typing import Any, List, Callable, Optional, Union

from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import TransformComponent, BaseNode

# Imports: GraphRAGStore
import re
from llama_index.core.graph_stores import SimplePropertyGraphStore
import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core.llms import ChatMessage

# Imports: GraphRAGQueryEngine
from llama_index.core.query_engine import CustomQueryEngine

# Imports: End-to-End GraphRAG Pipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PropertyGraphIndex

print("Imports working successfully")
nest_asyncio.apply()

# Load Data (bad, bad idea)
news = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv")[:50]
documents = [Document(text=f"{row['title']}: {row['text']}") for index, row in news.iterrows()]
print("Loaded Data")

# Setup API Key and LLM
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
az_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
az_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
llm = AzureOpenAI(engine="gpt-4o", model="gpt-4o", temperature=0.0, api_key=az_openai_api_key,
                  azure_endpoint=az_openai_endpoint)
print("Configurations up")

# GraphRAGExtractor
"""
GraphRAGExtractor class is designed to extract triples (subject-relation-object) from text and enrich them by adding 
descriptions for entities and relationships to their properties using an LLM.

Key Components
* llm: The language model used for extraction.
* extract_prompt: A prompt template used to guide the LLM in extracting information.
* parse_fn: A function to parse the LLM's output into structured data.
* max_paths_per_chunk: Limits the number of triples extracted per text chunk.
* num_workers: For parallel processing of multiple text nodes.

Main Methods
* __call__: The entry point for processing a list of text nodes.
* acall: An asynchronous version of call for improved performance.
* _aextract: The core method that processes each individual node.

Extraction Process
* For each chunk (node), sends text to LLM with the extraction prompt.
* LLM response is parsed to extract entities, relationships and descriptions for entities and relationships.
* Entities are converted into EntityNode objects, entity description stored in metadata.
* Relationships are converted into Relation objects. Relationship description is stored in metadata.
* All of these are added to node's metadata under KG_NODES_KEY and KG_RELATIONS_KEY.

Note
* Need to use entity description, only relationship descriptions are used here.
"""


class GraphRAGExtractor(TransformComponent):
    """
    Extract triples from a graph. Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples)
    and entity, relation descriptions from text.
    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
            self,
            llm: Optional[LLM] = None,
            extract_prompt: Optional[Union[str, PromptTemplate]] = None,
            parse_fn: Callable = default_parse_triplets_fn,
            max_paths_per_chunk: int = 10,
            num_workers: int = 4,
    ) -> None:
        """Init parameters."""
        from llama_index.core import Settings
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphRAGExtractor"

    def __call__(
            self, nodes: List[BaseNode], show_progress: bool = False, **kwargs
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        print("in _aextract")
        """Extract triples from a node"""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_graphs=self.max_paths_per_chunk
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            metadata["entity_description"] = description
            entity_node = EntityNode(name=entity, label=entity_type, properties=metadata)
            existing_nodes.append(entity_node)

        metadata = node.metadat.cope()
        for triple in entities_relationship:
            subj, rel, obj, description = triple
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        print("in _aextract: done")
        return node

    async def acall(self, nodes: List[BaseNode], show_progress: bool = False,
                    **kwargs: Any) -> List[BaseNode]:
        """Extract triples from nodes async"""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting triples from text"
        )


# GraphRAGStore
"""
The GraphRAGStore class is an extension of SimplePropertyGraphStore class, to implement GraphRAG pipeline.
The class uses community detection algorithms to group related nodes in the graph and generates summaries for each
community using an LLM.

Key Methods

* build_communities():
    1. Converts the internal graph representation to a NetworkX graph.
    2. Applies the hierarchical Leiden algorithm for community detection.
    3. Collects detailed information about each community.
    4. Generates summaries for each community.
* generate_community_summary(text):
    1. Uses LLM to generate a summary of the relationships in a community.
    2. The summary includes entity names and a synthesis of relationship descriptions.
* _create_nx_graph():
    1. Converts the internal graph representation to a NetworkX graph for community detection.
* _collect_community_info(nx_graph, clusters):
    1. Collects detailed information about each node based on its community.
    2. Creates a string representation of each relationship within a community.
* _summarize_communities(community_info):
    1. Generates and stores summaries for each community using LLM.
* get_community_summaries():
    1. Returns the community summaries by building them if not already done.
"""


class GraphRAGStore(SimplePropertyGraphStore):
    community_summary = {}
    max_cluster_size = 5

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM"""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise"
                    "synthesis of the relationship descriptions. The goal is to capture the most critical and relevant "
                    "details that highlight the nature and significance of each relationship. Ensure that the summary "
                    "is coherent and integrates the information in a way that emphasizes the key aspects of the"
                    "relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = AzureOpenAI.chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        print("generate community summary done")
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(nx_graph, max_cluster_size=self.max_cluster_size)
        community_info = self._collect_community_info(nx_graph, community_hierarchical_clusters)
        print("built communities")
        self._summarize_communities(community_info)
        print("summarized communities")

    def _create_nx_graph(self):
        """Converts internal graph representations to NetworkX graph."""
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relation_description"]
            )
        print("nx graph created")
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """Collect detailed information for each node based on their community"""
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
        print("community info collected")
        return community_info

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        print("Community summaries collected")
        return self.community_summary


# GraphRAGQueryEngine
"""
The GraphRAGQueryEngine class is a custom query engine designed to process queries using the GraphRAG approach. It
leverages the community summaries generated by the GraphRAGStore to answer user queries. 

Main Components
* graph_store: An instance of the GraphRAGStore, which contains the community summaries.
* llm: A language model used for generating and aggregating answers.

Key Methods
* custom_query(query_str: str)
    a. Entry point for processing a query. Retrieves community summaries, generates answers
    from each summary, and then aggregates these answers into a final response.
* generate_answer_from_summary(community_summary, query): 
    a. Generates an answer for the query based on a single community summary. Uses the LLM to interpret the community 
    summary in the context of the query.
* aggregate_answers(community_answers):
    a. Combines individual answers from different communities into a coherent final response.
    b. Uses the LLM to synthesize multiple perspectives into a single, concise answer.
    
Query Processing Flow
1. Retrieve community summaries from the graph store.
2. For each community summary, generate a specific answer to the query.
3. Aggregate all community-specific answers into a final, coherent response.
"""


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for _, community_summary in community_summaries.items()
        ]
        final_answer = self.aggregate_answers(community_answers)
        print("custom query answer generated")
        return final_answer

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary},"
            f"how would you answer the following query?"
            f"Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content="I need an answer based on the above information.")
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        print("answer generated from summary")
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=f"Intermediate answers: {community_answers}")
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(r"^assistant:\s*", "", str(final_response)).strip()
        print("answers aggregated")
        return cleaned_final_response


# End-to-End GraphRAG Pipeline
"""
Constructing the GraphRAG pipeline:
1. Create nodes/chunks from the text.
2. Build a PropertyGraphIndex using GraphRAGExtractor and GraphRAGStore.
3. Construct communities and generate a summary for each community using the graph built above.
4. Create a GraphRAGQueryEngine and begin querying.
"""

# Step 1: Create nodes/chunks from the text

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)
print("nodes created")


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

entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'


def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships


kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=2,
    parse_fn=parse_fn,
)
print("kg_extractor done")

index = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=GraphRAGStore(),
    kg_extractors=[kg_extractor],
    show_progress=True,
)
print("index done")
print(list(index.property_graph_store.graph.relations.values())[-1])

index.property_graph_store.build_communities()
print("communities built")
query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store, llm=llm
)
print("query engine done")

response = query_engine.query(
    "What are the main news discussed in the document?"
)
print(response)
