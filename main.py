import os
import hashlib
import logging
from pathlib import Path

import chromadb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ===== Локальная модель для эмбеддингов (замена DeepSeek) =====
from sentence_transformers import SentenceTransformer

# ===== Загрузка переменных окружения =====
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"

# ===== Логирование =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Системный промпт (роль Андрея) =====
SYSTEM_PROMPT = """# РОЛЬ
Ты — Андрей, профессиональный менеджер по входящим обращениям компании ООО «ЭНЕРГОИНЖИНИРИНГ».
Твоя задача — квалифицировать лида: задать 5–6 вопросов, собрать контакт, оформить заявку и передать в работу.

# ЯЗЫК
Отвечай на языке собеседника.

# СТИЛЬ
Профессионально, дружелюбно, коротко.
Правила:
- Один вопрос за сообщение.
- 1–2 предложения.
- Не повторяй ответ клиента «своими словами».
- Не выдумывай факты. Если чего-то не знаешь — используй ТОЛЬКО контекст ниже или переключай на менеджера.

# ПРИВЕТСТВИЕ
Если это первое сообщение в диалоге, поздоровайся, представься и назови компанию:
«Здравствуйте! Я Андрей, помощник электролаборатории «ЭНЕРГОИНЖИНИРИНГ». Быстро задам несколько вопросов, чтобы зафиксировать вашу заявку и подобрать решение.»

# ЛОГИКА ДИАЛОГА (пошагово, линейно)
Шаг 1. Приветствие.
Шаг 2. Вопрос: «Что вас интересует: электроиспытания и измерения, электромонтажные работы, техническое обслуживание, энергоаудит или консультация по документации?»
Шаг 3. Вопрос: «По какому именно виду работ нужна помощь? Например: измерение сопротивления изоляции, проверка заземления, испытания автоматов, петля «фаза-ноль» или что-то другое?»
Шаг 4. Вопрос: «Уточните, пожалуйста: какой у вас объект (тип помещения, примерная площадь) и в каком районе Москвы или Московской области он находится?»
Шаг 5. Вопрос: «Когда вам нужен выезд специалиста — в день обращения, на этой неделе или к конкретной дате?»
Шаг 6. Вопрос: «Оставьте, пожалуйста, ваш номер телефона. Наш инженер свяжется с вами, уточнит детали и согласует удобное время выезда.»

# ОТВЕТЫ НА ТЕХНИЧЕСКИЕ ВОПРОСЫ
Если клиент задаёт технический вопрос (нормы, методики, приборы, сроки, документы) — отвечай КОРОТКО на основе КОНТЕКСТА ниже.
После ответа возвращайся к логике диалога и задавай следующий вопрос по шагам.
Если в контексте нет ответа — скажи: «Уточню у старшего инженера, он свяжется с вами в течение 15 минут.»

# ПЕРЕКЛЮЧЕНИЕ НА МЕНЕДЖЕРА
Переключай на старшего менеджера, если:
- Клиент прямо просит человека/оператора.
- Клиент раздражён или отказывается продолжать.
- Клиент задаёт сложный технический или юридический вопрос, на который нет однозначного короткого ответа.
Тогда ответь: «Понял вас. Чтобы решить этот вопрос максимально точно, я прямо сейчас передам вашу заявку старшему инженеру, и он свяжется с вами в течение 15 минут.»"""

# ===== FastAPI =====
app = FastAPI(title="Energoengineering RAG Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ChromaDB =====
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="knowledge")

# ===== Локальная модель эмбеддингов =====
# Мультиязычная, лёгкая (~120 МБ), работает на CPU
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ===== Модели данных =====
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# ===== Хэш файла знаний =====
def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()

# ===== Эмбеддинги (локально, без API) =====
def get_embedding(text: str) -> list:
    """Получает векторное представление текста локальной моделью."""
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# ===== Загрузка базы знаний =====
def load_knowledge():
    global collection

    knowledge_file = Path("knowledge.txt")
    if not knowledge_file.exists():
        logger.warning("knowledge.txt не найден. База будет пустой.")
        return

    current_hash = file_hash(knowledge_file)
    hash_file = Path(CHROMA_PATH) / "knowledge.hash"

    if hash_file.exists() and hash_file.read_text().strip() == current_hash:
        logger.info(f"База актуальна: {collection.count()} чанков")
        return

    logger.info("Обновляю базу знаний...")
    try:
        client.delete_collection("knowledge")
    except Exception:
        pass

    collection = client.get_or_create_collection(name="knowledge")

    content = knowledge_file.read_text(encoding="utf-8")
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"chunk_{i}"],
            )
        except Exception as e:
            logger.error(f"Ошибка при загрузке чанка {i}: {e}")

    hash_file.write_text(current_hash)
    logger.info(f"Загружено {len(chunks)} чанков.")

# ===== Поиск в базе =====
def search_knowledge(question: str, n_results: int = 4) -> str:
    if collection.count() == 0:
        return ""
    try:
        query_embedding = get_embedding(question)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
        )
        if not results["documents"] or not results["documents"][0]:
            return ""
        return "\n\n".join(results["documents"][0])
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        return ""

# ===== Эндпоинты =====
@app.on_event("startup")
async def startup():
    load_knowledge()

@app.get("/health")
async def health():
    return {"status": "ok", "chunks": collection.count()}

@app.post("/reload")
async def reload_knowledge():
    load_knowledge()
    return {"status": "reloaded", "chunks": collection.count()}

@app.post("/ask")
async def ask(req: ChatRequest):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")

    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages")

    last_user_message = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user_message = m.content
            break

    if not last_user_message.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        context = search_knowledge(last_user_message)

        deepseek_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            deepseek_messages.append({
                "role": "system",
                "content": f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context}"
            })

        for m in req.messages:
            deepseek_messages.append({"role": m.role, "content": m.content})

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-chat",
            "messages": deepseek_messages,
            "temperature": 0.3,
            "max_tokens": 600,
        }

        response = requests.post(
            DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]

        return {"answer": answer}

    except requests.RequestException as e:
        logger.error(f"Network error: {e}")
        raise HTTPException(status_code=502, detail="Ошибка соединения с API")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
