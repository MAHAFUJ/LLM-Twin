from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

class AdvancedRAG:
    def __init__(self, collection_name: str = "llm_twin_context"):
        self.client = QdrantClient(":memory:") 
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.collection_name = collection_name

    def expand_query(self, query: str) -> list[str]:
        return [query, f"Details regarding {query}", f"My perspective on {query}"]

    def retrieve_and_rerank(self, query: str, top_k: int = 3) -> list[dict]:
        queries = self.expand_query(query)
        all_results = []
        
        for q in queries:
            vector = self.embed_model.encode(q).tolist()
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=5
            )
            all_results.extend([hit.payload for hit in hits])

        unique_results = {res["text"]: res for res in all_results}.values()
        
        pairs = [[query, res["text"]] for res in unique_results]
        scores = self.reranker.predict(pairs)
        
        ranked = sorted(zip(unique_results, scores), key=lambda x: x[1], reverse=True)
        return [res for res, score in ranked[:top_k]]

    def generate_response(self, query: str) -> str:
        context_docs = self.retrieve_and_rerank(query)
        context_str = "\n".join([doc["text"] for doc in context_docs])
        
        prompt = f"""Use the following context to answer like the user's AI twin.
Context: {context_str}
Query: {query}
Answer:"""
        return prompt