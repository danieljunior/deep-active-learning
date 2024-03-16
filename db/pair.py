from base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, Float, ForeignKey

class Pair(Base):
    __tablename__ = "pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document1_id = mapped_column(ForeignKey("documents.id"), nullable=False)
    document1 = relationship("Document", foreign_keys='Pair.document1_id')
    document2_id = mapped_column(ForeignKey("documents.id"), nullable=False)
    document2 = relationship("Document", foreign_keys='Pair.document2_id')
    label = mapped_column(Float)
    sampling = relationship("Sampling", back_populates="pair")


def __repr__(self) -> str:
        return f"Pair(id={self.id!r}, doc1={self.document1.id!r}, doc2={self.document2.id!r})"
