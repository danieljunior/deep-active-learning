from base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, Integer, String
from typing import List

class Run(Base):
    __tablename__ = "runs"

    id = mapped_column(Integer, primary_key=True)
    strategy_id = mapped_column(ForeignKey("strategies.id"), nullable=False)
    query_size = mapped_column(Integer, nullable=False)
    pool_size = mapped_column(Integer, nullable=False)
    n_rounds = mapped_column(Integer, nullable=False)
    seed = mapped_column(Integer, nullable=False)
    model_id = mapped_column(ForeignKey("models.id"), nullable=False)
    metric = mapped_column(String, nullable=False)

    strategy = relationship("Strategy", back_populates="runs")
    model = relationship("Model", back_populates="runs")
    rounds: Mapped[List["Round"]] = relationship(back_populates="run")

    def __repr__(self) -> str:
        return f"Run(id={self.id!r}, strategy={self.strategy!r}"