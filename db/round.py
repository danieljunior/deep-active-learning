from base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Double, ForeignKey

class Round(Base):
    __tablename__ = "rounds"

    id = mapped_column(Integer, primary_key=True)
    run_id = mapped_column(ForeignKey("runs.id"))
    run = relationship("Run", back_populates="rounds")
    samplings = relationship("Sampling", back_populates="round")
    iteration = mapped_column(Integer)
    perfomance = mapped_column(Double)

    def __repr__(self) -> str:
        return f"Run(id={self.id!r}, strategy={self.strategy!r}, iteration={self.iteration!r})"
