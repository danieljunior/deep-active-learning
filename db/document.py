from base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Text, String, Integer
from pgvector.sqlalchemy import Vector
from typing import List

class Document(Base):
    __tablename__ = "documents"

    id = mapped_column(Integer, primary_key=True)
    text = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(3), nullable=False)
    hash = mapped_column(String, nullable=False)

    # pairs: Mapped[List["Pair"]] = relationship(back_populates="document")

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, text={self.text!r})"
