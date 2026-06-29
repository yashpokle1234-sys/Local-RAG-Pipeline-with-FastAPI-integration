import os
import json 
import hashlib
from fastapi import FastAPI, HTTPException
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pydantic import Field
from pydantic import BaseModel
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import redis
import httpx

from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.llm.providers import register_llm_provider
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, AIMessage,SystemMessage,HumanMessage

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class StableOllamaProvider(BaseChatModel):
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
            content = response.json()['message']['content']
        
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "stable_ollama_chat"

register_llm_provider("stable_ollama", StableOllamaProvider)

QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
COLLECTION_NAME = 'knowledge_base'

rails_config = RailsConfig.from_path('./config')
rails_runtime = LLMRails(config=rails_config)

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app = FastAPI(title='Private RAG Application Gateway')
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
inference_brain = StableOllamaProvider()

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, strip_whitespace=True)


@app.post('/api/chat')
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    request = body:
    try:
        user_query = request.question
        cache_key = f"query:{hashlib.md5(user_query.strip().lower().encode()).hexdigest()}"
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Cache hit for query: {user_query[:50]}")
            return json.loads(cached)

        guardrail_response = await rails_runtime.generate_async(
            messages=[{'role' : 'user', 'content' : user_query}])
        guardrail_text = guardrail_response.get('content :' , '')
        if not guardrail_text or guardrail_response.get('stop_reason') == 'rails':
            return {
                "status": "guarded_refusal",
                "answer": guardrail_text or 'this query is outside of the scope',
                "chunks_retrieved": 0
            }

        query_vector = list(model.embed([user_query]))[0].tolist()
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        retrieved_contexts = [point.payload['text'] for point in search_results if point.payload]
        context_payload = "\n---\n".join(retrieved_contexts) if retrieved_contexts else "context not found"
        prompt_messages = [
            SystemMessage(content=f"You are a helpful private assistant answer the user's question using only the context below.if the answer is not in the cintext ,say u dont know.\n{context_payload}"),
            HumanMessage(content=user_query)
        ]
        response = inference_brain._generate(messages=prompt_messages)
        
        return {
            'status': 'success',
            'answer': response.generations[0].message.content,
            'chunks_retrieved': len(retrieved_contexts)
        }
        redis_client.setex(cache_key, 3600, json.dumps(result))
        logger.info(f"Cached result for query: {user_query[:50]}")
        return result
    except httpx.TimeoutException:
        logger.error("Ollama request timed out")
        raise HTTPException(status_code=504, detail="LLM timed out — try again")

    except httpx.ConnectError:
        logger.error("Could not connect to Ollama")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    except Exception as e:
        logger.error(f"Unexpected error in /api/chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get('/health')
async def health_check():
    return {
        "status": "healthy", 
        "storage_indexed": qdrant_client.collection_exists(COLLECTION_NAME)
    }