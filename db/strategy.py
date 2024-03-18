from base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Text, Integer, String
from typing import List

class Strategy(Base):
    __tablename__ = "strategies"

    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String, nullable=False)
    description = mapped_column(Text, nullable=False)

    runs: Mapped[List["Run"]] = relationship(back_populates="strategy")

    def __repr__(self) -> str:
        return f"Strategy(id={self.id!r}, name={self.name!r})"
