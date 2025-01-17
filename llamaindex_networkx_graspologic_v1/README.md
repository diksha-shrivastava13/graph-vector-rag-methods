## GraphRAG Implementation with LlamaIndex, Graspologic and NetworkX

GraphRAG combines the strengths of Retrieval Augmented Generation (RAG) and Query-Focused
Summarization (QFS) to effectively handle complex queries over large text datasets.

* RAG excels in fetching precise information, struggles with broader queries that require
thematic understanding.
* QFS can tackle issues that require thematic understanding but cannot scale well.

GraphRAG integrates these two to tackle thematic problem understanding and can scale well.

Steps to Build GraphRAG:
```
1. Graph Generation:
    a. Source Documents to Text Chunks              - Implemented with SentenceSplitter
    b. Text Chunks to Element Summaries             - Implemented with GraphRAGExtractor
    c. Element Instances to Element Summaries       - Implemented with GraphRAGExtractor
    d. Element Summaries to Graph Communities       - implemented using GraphRAGStore
    e. Graph Communities to Community Summaries     - Implemented with GraphRAGStore
2. Answering the Query:
    a. Community Summaries to Global Answers.       - Implemented with GraphQueryEngine
```

### GraphRAGExtractor

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


### GraphRAGStore

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


### GraphRAGQueryEngine

The GraphRAGQueryEngine class is a custom query engine designed to process queries using the GraphRAG approach. It
leverages the community summaries generated by the GraphRAGStore to answer user queries. 

Main Components
* graph_store: An instance of the GraphRAGStore, which contains the community summaries.
* llm: A language model used for generating and aggregating answers.

Key Methods
```
* custom_query(query_str: str)
    a. Entry point for processing a query. Retrieves community summaries, generates answers
    from each summary, and then aggregates these answers into a final response.
* generate_answer_from_summary(community_summary, query): 
    a. Generates an answer for the query based on a single community summary. Uses the LLM to interpret the community 
    summary in the context of the query.
* aggregate_answers(community_answers):
    a. Combines individual answers from different communities into a coherent final response.
    b. Uses the LLM to synthesize multiple perspectives into a single, concise answer.
```
Query Processing Flow
1. Retrieve community summaries from the graph store.
2. For each community summary, generate a specific answer to the query.
3. Aggregate all community-specific answers into a final, coherent response.


### End-to-End GraphRAG Pipeline

Constructing the GraphRAG pipeline:
1. Create nodes/chunks from the text.
2. Build a PropertyGraphIndex using GraphRAGExtractor and GraphRAGStore.
3. Construct communities and generate a summary for each community using the graph built above.
4. Create a GraphRAGQueryEngine and begin querying.
