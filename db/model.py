from base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Text, Integer, String, JSON
from typing import List

class Model(Base):
    __tablename__ = "models"

    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String, nullable=False)
    hf_checkpoint = mapped_column(String, nullable=False)
    train_configs = mapped_column(JSON, nullable=False)

    runs: Mapped[List["Run"]] = relationship(back_populates="model")

    def __repr__(self) -> str:
        return f"Model(id={self.id!r}, name={self.name!r})"
