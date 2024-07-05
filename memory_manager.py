import uuid
import chromadb
from chromadb.utils import embedding_functions

class MemoryManager:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.memory_collection = self.chroma_client.create_collection(
            name="artmuse_memory",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
        )

    def add_to_memory(self, role, content):
        self.memory_collection.add(
            documents=[content],
            metadatas=[{"role": role}],
            ids=[str(uuid.uuid4())]
        )

    def get_relevant_memory(self, query, n_results=5):
        results = self.memory_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        memories = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            memories.append(f"{metadata['role']}: {doc}")
        return "\n".join(memories)