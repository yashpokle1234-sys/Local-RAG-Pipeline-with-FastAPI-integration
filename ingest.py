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
        print('no documents found in the directory')
        return
        
    chunks = chunk_documents(docs)
    print(f'chunks = {len(chunks)} ')

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_size = 384

    # qdrant client setting
    if qdrant_client.collectiom_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)



    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
    points = []
    for idx, chunk in enumerate(chunks):
        embedding = list(model.embed([chunk['text']]))[0].tolist()

        points.append(
            PointStruct(id=idx,vector=embedding,payload={'text': chunk['text'],'metadata': chunk['metadata']}))
                    
        # cache preview backup in =  Redis 
        redis_key = f'doc:chunk:{idx}'
        redis_client.hset(redis_key, mapping={
            'source': chunk['metadata']['source'],
            'text_preview': chunk['text'][:100]
        })

    # store into qdrant instance
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    


if __name__ == '__main__':
    main()