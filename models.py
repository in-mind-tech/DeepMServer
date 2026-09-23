from sqlalchemy import Column, BigInteger, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from database import Base


class Scan(Base):
    __tablename__ = "scan"

    id = Column(
        BigInteger,
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
        JSONB,
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