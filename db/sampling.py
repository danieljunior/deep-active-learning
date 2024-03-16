from base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Text, ForeignKey, Integer, Double

class Sampling(Base):
    __tablename__ = "samplings"

    id = mapped_column(Integer, primary_key=True)
    round_id = mapped_column(ForeignKey("rounds.id"))
    round = relationship("Round", back_populates="samplings")
    pair_id = mapped_column(ForeignKey("pairs.id"))
    pair = relationship("Pair", back_populates="sampling")
    criteria_value = mapped_column(Double)

    def __repr__(self) -> str:
        return f"Sampling(id={self.id!r})"
