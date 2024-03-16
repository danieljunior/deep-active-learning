from base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Text
from pgvector.sqlalchemy import Vector
from typing import List

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(3))

    # pairs: Mapped[List["Pair"]] = relationship(back_populates="document")

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, text={self.text!r})"
