import os 
import sys
import uuid
import logging
import time
import glob
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding 
import redis
from langchain_text_splitters import RecursiveCharacterTextSplitter

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

COLLECTION_NAME = 'knowledge_base'

def initialize_clients():
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return qdrant_client, redis_client

def load_docs(directory_path='./data'):
    documents = []
    search_path = os.path.join(directory_path, '*.txt')
    for file_path in glob.glob(search_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append({
                'source': os.path.basename(file_path),
                'text': f.read()
            })
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        texts = splitter.split_text(doc['text'])
        for i, text in enumerate(texts):
            chunks.append({
                'text': text,
                'metadata': {
                    'source': doc['source'],
                    'chunk_index': i
                }
            })
    return chunks

def main():
    qdrant_client, redis_client = initialize_clients()
    docs = load_docs()
    if not docs:
        print('No documents found in ./data directory.')
        return
        
    chunks = chunk_documents(docs)
    print(f'Created {len(chunks)} text chunks from source documents.')

    # Using the low-overhead CPU embedding manager that matches app.py
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # BAAI/bge-small-en-v1.5 uses 384 dimensions
    vector_size = 384

    # Setting up the clean vector collection matrix
    print(f'Setting up collection: "{COLLECTION_NAME}" in Qdrant Vector DB...')
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    points = []
    for idx, chunk in enumerate(chunks):
        # Generate the vector embedding using fastembed
        embedding = list(model.embed([chunk['text']]))[0].tolist()

        # Build Qdrant document point struct payload
        points.append(
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    'text': chunk['text'],
                    'metadata': chunk['metadata']
                }
            )
        )
                    
        # Seed cache preview backup into Redis hset mapping hooks
        redis_key = f'doc:chunk:{idx}'
        redis_client.hset(redis_key, mapping={
            'source': chunk['metadata']['source'],
            'text_preview': chunk['text'][:100]
        })

    # Upload everything into your running Qdrant instance
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Ingestion successfully completed! All context vectors stored in Qdrant database.")

if __name__ == '__main__':
    main()