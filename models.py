from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Integer,
    DateTime,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from database import Base


# ============================================================
# TABLE SCAN
# Données temporaires envoyées par les téléphones
# ============================================================

class Scan(Base):
    __tablename__ = "scan"

    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    idprofile = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    discussion_data = Column(
        JSONB,
        nullable=False,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ============================================================
# TABLE MATCH_RESULT
#
# Résultat temporaire d'un match.
#
# Elle permet :
# - au scanner de récupérer le résultat
# - au propriétaire du QR de récupérer le même résultat
# - d'inverser les deux pourcentages côté application
#
# IMPORTANT :
# aucune conversation Cupid complète n'est enregistrée ici.
# ============================================================

class MatchResult(Base):
    __tablename__ = "match_result"

    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    match_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    profile_a_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    profile_b_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    # A correspond aux attentes de B
    match_a_to_b = Column(
        Integer,
        nullable=False,
    )

    # B correspond aux attentes de A
    match_b_to_a = Column(
        Integer,
        nullable=False,
    )

    # Ex:
    # [
    #   "religion",
    #   "sexuality",
    #   "conflict_management"
    # ]
    tensions = Column(
        JSONB,
        nullable=False,
        default=list,
    )

    # Résumé général non sensible
    summary = Column(
        String(2000),
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


# ============================================================
# TABLE HISTORIQUE
#
# Statistiques limitées.
#
# Pas de :
# - idprofile
# - réponses Cupid
# - religion précise
# - réponses sexuelles détaillées
# - texte complet de conversation
# ============================================================

class Historique(Base):
    __tablename__ = "historique"

    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    my_gender = Column(
        String(30),
        nullable=True,
    )

    scanned_gender = Column(
        String(30),
        nullable=True,
    )

    my_orientation = Column(
        String(50),
        nullable=True,
    )

    scanned_orientation = Column(
        String(50),
        nullable=True,
    )

    my_country = Column(
        String(100),
        nullable=True,
    )

    scanned_country = Column(
        String(100),
        nullable=True,
    )

    # Mon profil -> attentes de l'autre
    match_my_to_their = Column(
        Integer,
        nullable=False,
    )

    # Son profil -> mes attentes
    match_their_to_my = Column(
        Integer,
        nullable=False,
    )

    # Ex:
    # ["religion", "conflict_management"]
    conflit = Column(
        JSONB,
        nullable=False,
        default=list,
    )

    datescan = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )