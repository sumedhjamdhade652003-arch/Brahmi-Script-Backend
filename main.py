import os
import json
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# =========================================================
# 1. Database Configuration (Local PostgreSQL)
# =========================================================
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Use psycopg driver explicitly
DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env file.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String(255), nullable=False, index=True)
    english = Column(Text, nullable=False)
    hindi = Column(Text, nullable=False)
    marathi = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================================================
# 2. Pydantic Schemas
# =========================================================
class WordTranslationRequest(BaseModel):
    word: str = Field(..., min_length=1, example="kamala", description="Brahmi transliterated word")

class TranslationResponse(BaseModel):
    id: int
    word: str
    english: str
    hindi: str
    marathi: str
    created_at: datetime

    class Config:
        from_attributes = True

class DeleteResponse(BaseModel):
    message: str
    deleted_id: Optional[int] = None
    deleted_count: Optional[int] = None

# =========================================================
# 3. Gemini Translation Service
# =========================================================
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

async def fetch_gemini_translation(word: str) -> dict:
    prompt = f"""
    You are an expert linguist specializing in ancient Indian scripts and languages.
    Translate and provide the accurate meaning for the Brahmi transliterated word/term: "{word}"
    into:
    1. English
    2. Hindi (Devanagari script)
    3. Marathi (Devanagari script)

    Output strictly a JSON object with this exact structure:
    {{
        "english": "English meaning",
        "hindi": "हिंदी अर्थ",
        "marathi": "मराठी अर्थ"
    }}
    Do not output markdown code blocks, backticks, or any conversational text.
    """

    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )
        )
        return json.loads(response.text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API translation failed: {str(exc)}"
        )

# =========================================================
# 4. FastAPI Application and Endpoints
# =========================================================
app = FastAPI(
    title="Brahmi Word Translation Engine",
    description="Translation API using Gemini with local PostgreSQL history.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Brahmi Translation Engine"}

# 1. Translate Word & Record History
@app.post("/api/translate", response_model=TranslationResponse, status_code=status.HTTP_201_CREATED)
async def translate_word(
    payload: WordTranslationRequest,
    db: Session = Depends(get_db)
):
    target_word = payload.word.strip()
    if not target_word:
        raise HTTPException(status_code=400, detail="Word cannot be blank.")

    translations = await fetch_gemini_translation(target_word)

    new_record = TranslationHistory(
        word=target_word,
        english=translations.get("english", ""),
        hindi=translations.get("hindi", ""),
        marathi=translations.get("marathi", "")
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    return new_record

# 2. View History
@app.get("/api/history", response_model=List[TranslationResponse])
def get_history(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    return (
        db.query(TranslationHistory)
        .order_by(TranslationHistory.created_at.desc())
        .limit(limit)
        .all()
    )

# 3. Delete Single Entry by ID
@app.delete("/api/history/{history_id}", response_model=DeleteResponse)
def delete_single_history(
    history_id: int,
    db: Session = Depends(get_db)
):
    record = db.query(TranslationHistory).filter(TranslationHistory.id == history_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"History record with ID {history_id} not found.")

    db.delete(record)
    db.commit()
    return DeleteResponse(message="Record deleted successfully", deleted_id=history_id)

# 4. Delete All Entries
@app.delete("/api/history", response_model=DeleteResponse)
def clear_all_history(
    db: Session = Depends(get_db)
):
    count = db.query(TranslationHistory).delete()
    db.commit()
    return DeleteResponse(message="All history cleared successfully", deleted_count=count)