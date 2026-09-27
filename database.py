import os

from sqlalchemy import create_engine
from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
)


# ============================================================
# DATABASE URL
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL n'est pas configurée. "
        "Ajoutez cette variable d'environnement "
        "dans les paramètres de votre service Render."
    )


# ============================================================
# FORCER LE DRIVER PSYCOPG (v3)
# ============================================================
#
# Render fournit une DATABASE_URL qui commence par :
#
#     postgresql://user:pass@host:port/db
#
# Mais SQLAlchemy doit savoir explicitement quel driver
# utiliser, car plusieurs drivers peuvent être installés.
#
# Sur Python 3.14, SQLAlchemy préfère "psycopg" (v3) à
# "psycopg2". Si on ne précise rien, il essaie d'importer
# "psycopg" et échoue si seul "psycopg2" est installé.
#
# On force donc "postgresql+psycopg://" pour utiliser
# le driver moderne psycopg (v3).
#
# ============================================================
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1,
    )

elif DATABASE_URL.startswith("postgres://"):
    # Certains hébergeurs utilisent l'ancien préfixe
    # "postgres://" (sans "ql"). On le normalise aussi.
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql+psycopg://",
        1,
    )


# ============================================================
# ENGINE SQLALCHEMY
# ============================================================
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # pool_pre_ping=True vérifie que la connexion est
    # encore vivante avant chaque utilisation. Utile
    # pour Render qui peut suspendre la base PostgreSQL
    # après une période d'inactivité.
)


# ============================================================
# SESSION
# ============================================================
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ============================================================
# BASE POUR LES MODÈLES
# ============================================================
Base = declarative_base()


# ============================================================
# DÉPENDANCE FASTAPI
# ============================================================
def get_db():
    """
    Fournit une session SQLAlchemy à chaque requête FastAPI.
    La session est automatiquement fermée après la requête.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()