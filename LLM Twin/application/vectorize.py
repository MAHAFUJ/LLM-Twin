import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class VectorizationPipeline:
    def __init__(self, collection_name: str = "llm_twin_context"):
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.client = QdrantClient(":memory:") 
        self.collection_name = collection_name
        
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def process_and_store(self, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        points = []

        for item in data:
            chunks = splitter.split_text(item["content"])
            for chunk in chunks:
                vector = self.embed_model.encode(chunk).tolist()
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={"text": chunk, "source": item["source"]}
                    )
                )
        
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Stored {len(points)} chunks in vector database.")

if __name__ == "__main__":
    pipeline = VectorizationPipeline()
    pipeline.process_and_store("data/raw/ingestion_20260722.json")