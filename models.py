from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Integer,
    DateTime,
    Boolean,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from database import Base

# ============================================================
# TABLE SCAN
# ============================================================
#
# Cette table contient temporairement les données de profil
# nécessaires au fonctionnement du Deep Matching.
#
# IMPORTANT :
#
# - Elle peut être alimentée lorsqu'un utilisateur affiche
# son QR Code.
#
# - Elle peut être mise à jour lorsque son profil change.
#
# - Elle sert au serveur pour effectuer une analyse.
#
# - Elle ne doit PAS être utilisée comme historique.
#
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
# ============================================================
#
# Cette table contient UNIQUEMENT le résultat temporaire
# d'un match entre deux utilisateurs.
#
# Elle n'est PAS une table historique.
#
# Elle sert uniquement à permettre :
#
# 1. au scanner de recevoir son résultat ;
# 2. au propriétaire du QR de recevoir son résultat ;
# 3. de conserver le même match pour les deux utilisateurs ;
# 4. d'inverser les scores selon l'utilisateur qui consulte.
#
# Une fois que les DEUX utilisateurs ont reçu leur résultat,
# le résultat peut être supprimé.
#
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

    match_a_to_b = Column(
        Integer,
        nullable=False,
    )

    match_b_to_a = Column(
        Integer,
        nullable=False,
    )

    tensions = Column(
        JSONB,
        nullable=False,
        default=list,
    )

    summary = Column(
        String(2000),
        nullable=True,
    )

    # ========================================================
    # ÉTAT DE RÉCEPTION
    # ========================================================
    #
    # Ces deux champs permettent au serveur de savoir si les
    # deux appareils ont déjà reçu leur résultat.
    #
    # False = résultat pas encore récupéré par cet utilisateur.
    # True = résultat déjà récupéré.
    #
    # Lorsque les deux deviennent True, main.py pourra supprimer
    # la ligne MatchResult.
    #
    # ========================================================
    scanner_received = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    owner_received = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

# ============================================================
# TABLE HISTORIQUE
# ============================================================
#
# CETTE TABLE EST RÉSERVÉE À L'ADMINISTRATEUR.
#
# Elle sert uniquement à l'analyse statistique du service.
#
# L'APPLICATION MOBILE NE DOIT PAS :
#
# - lire cette table ;
# - récupérer ses données ;
# - afficher ses données ;
# - modifier ses données ;
# - supprimer ses données ;
# - utiliser cette table pour retrouver un ancien résultat.
#
# Le serveur peut toutefois INSÉRER une ligne lorsqu'une
# analyse est réalisée.
#
# ============================================================
#
# IMPORTANT :
#
# Aucun idprofile n'est enregistré.
#
# Aucune réponse Cupid complète n'est enregistrée.
#
# Aucun texte privé n'est enregistré.
#
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

    match_my_to_their = Column(
        Integer,
        nullable=False,
    )

    match_their_to_my = Column(
        Integer,
        nullable=False,
    )

    conflit = Column(
        JSONB,
        nullable=False,
        default=list,
    )

    datescan = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )