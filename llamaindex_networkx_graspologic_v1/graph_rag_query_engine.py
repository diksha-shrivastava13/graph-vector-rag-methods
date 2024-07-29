import re
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.query_engine import CustomQueryEngine
from graph_rag_store import GraphRAGStore


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
        return cleaned_final_response
