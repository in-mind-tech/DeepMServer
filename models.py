from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func

from database import Base


class Scan(Base):
    __tablename__ = "scan"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    idprofile = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )

    discussion_data = Column(
        JSON,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )