import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import redis
import httpx

# NeMo Guardrails Native Custom Registration Imports
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.llm.providers import register_llm_provider
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, AIMessage

# =====================================================================
# 1. CUSTOM OLLAMA NATIVE PROVIDER (Bypasses picky 404 OpenAI endpoints)
# =====================================================================
class StableOllamaProvider(BaseChatModel):
    """Custom provider communicating directly with Ollama's native chat route."""
    api_url: str = 'http://ollama:11434/api/chat'
    model_name: str = 'llama3'

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload_messages = []
        for msg in messages:
            role = "user"
            if "system" in str(type(msg)).lower():
                role = "system"
            elif "ai" in str(type(msg)).lower():
                role = "assistant"
            payload_messages.append({"role": role, "content": msg.content})

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self.api_url,
                json={"model": self.model_name, "messages": payload_messages, "stream": False},
            )
            response.raise_for_status()
            # Parse native Ollama API response object schema
            content = response.json()['message']['content']
        
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "stable_ollama_chat"

# Register our safe custom type globally inside NeMo's memory map
register_llm_provider("stable_ollama", StableOllamaProvider)

# =====================================================================
# 2. CONFIG CONNECTIONS & RUNTIME
# =====================================================================
QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
COLLECTION_NAME = 'knowledge_base'

# Load NeMo Guardrails configuration profile
rails_config = RailsConfig.from_path('./config')
rails_runtime = LLMRails(config=rails_config)

# Standard App Setup
app = FastAPI(title='Private RAG Application Gateway')
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Highly optimized CPU embeddings to protect your laptop thermals
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
inference_brain = StableOllamaProvider()

class ChatRequest(BaseModel):
    question: str

# =====================================================================
# 3. ENDPOINTS
# =====================================================================
@app.post('/api/chat')
async def chat_endpoint(request: ChatRequest):
    try:
        user_query = request.question

        # Query checked via guardrails pipeline
        guardrail_response = await rails_runtime.generate_async(prompt=user_query)
        if "I am programmed to assist exclusively" in guardrail_response:
            return {
                "status": "guarded_refusal",
                "answer": guardrail_response,
                "chunks_retrieved": 0
            }

        # Vector conversion via fastembed
        query_vector = list(model.embed([user_query]))[0].tolist()
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        retrieved_contexts = [point.payload['text'] for point in search_results if point.payload]
        context_payload = "\n---\n".join(retrieved_contexts) if retrieved_contexts else "No context found."

        # Pass context natively through our stable custom class
        prompt_messages = [
            AIMessage(content=f"Answer using only this context:\n{context_payload}"),
            AIMessage(content=user_query)
        ]
        response = inference_brain._generate(messages=prompt_messages)
        
        return {
            'status': 'success',
            'answer': response.generations[0].message.content,
            'chunks_retrieved': len(retrieved_contexts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health_check():
    return {
        "status": "healthy", 
        "storage_indexed": qdrant_client.collection_exists(COLLECTION_NAME)
    }