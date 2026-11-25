import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DB_PATH = Path(__file__).with_name("outfit_feedback.sqlite3")


class OutfitFeedback(BaseModel):
    clothing_request: str = Field(min_length=1, max_length=500)
    outfit: str = Field(min_length=1, max_length=200)
    face_shape: str = Field(max_length=30)
    body_shape: str = Field(max_length=30)
    skin_depth: str = Field(max_length=30)
    rating: Literal["like", "dislike"]


def save_feedback(feedback: OutfitFeedback) -> None:
    with closing(sqlite3.connect(DB_PATH)) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS outfit_feedback (
                id INTEGER PRIMARY KEY,
                clothing_request TEXT NOT NULL,
                outfit TEXT NOT NULL,
                face_shape TEXT NOT NULL,
                body_shape TEXT NOT NULL,
                skin_depth TEXT NOT NULL,
                rating TEXT NOT NULL
            )
        """)
        db.execute("""
            INSERT INTO outfit_feedback
                (clothing_request, outfit, face_shape, body_shape, skin_depth, rating)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            feedback.clothing_request, feedback.outfit, feedback.face_shape,
            feedback.body_shape, feedback.skin_depth, feedback.rating,
        ))