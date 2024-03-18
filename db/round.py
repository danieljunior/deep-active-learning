from base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Double, ForeignKey

class Round(Base):
    __tablename__ = "rounds"

    id = mapped_column(Integer, primary_key=True)
    run_id = mapped_column(ForeignKey("runs.id"), nullable=False)
    iteration = mapped_column(Integer, nullable=False)
    performance = mapped_column(Double, nullable=False)
    loss = mapped_column(Double, nullable=False)

    samplings = relationship("Sampling", back_populates="round")
    run = relationship("Run", back_populates="rounds")

    def __repr__(self) -> str:
        return f"Run(id={self.id!r}, strategy={self.strategy!r}, iteration={self.iteration!r})"
