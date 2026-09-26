import json
import os

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Dict, Any
from groq import Groq

from database import Base, engine, get_db
from models import Scan


# ============================================================
# DATABASE
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="Deep Matching API",
    description="API Backend de l'application Deep Matching - Cupid V3",
    version="3.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = None

if GROQ_API_KEY:
    groq_client = Groq(
        api_key=GROQ_API_KEY,
    )


# ============================================================
# MODELES PYDANTIC
# ============================================================

class ScanCreate(BaseModel):
    idprofile: str
    discussion_data: Dict[str, Any]


class ScanUpdate(BaseModel):
    discussion_data: Dict[str, Any]


class MatchRequest(BaseModel):
    my_profile_id: str
    scanned_profile_id: str
    language: str = "fr"


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "application": "Deep Matching",
        "service": "Cupid V3 API",
        "version": "3.1.0",
        "status": "online",
        "ai": "Groq",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "database": "PostgreSQL",
        "service": "deep-matching-api",
        "groq_configured": bool(GROQ_API_KEY),
    }


# ============================================================
# CREATE SCAN
# ============================================================

@app.post(
    "/api/v1/scan",
    status_code=status.HTTP_201_CREATED,
)
def create_scan(
    data: ScanCreate,
    db: Session = Depends(get_db),
):
    profile_id = data.idprofile.strip()

    if not profile_id:
        raise HTTPException(
            status_code=400,
            detail="Identifiant du profil invalide",
        )

    existing_scan = (
        db.query(Scan)
        .filter(
            Scan.idprofile == profile_id
        )
        .first()
    )

    if existing_scan:
        raise HTTPException(
            status_code=409,
            detail="Ce profil existe déjà",
        )

    scan = Scan(
        idprofile=profile_id,
        discussion_data=data.discussion_data,
    )

    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {
        "success": True,
        "message": "Profil enregistré",
        "id": scan.id,
        "idprofile": scan.idprofile,
    }


# ============================================================
# GET SCAN
# ============================================================

@app.get("/api/v1/scan/{idprofile}")
def get_scan(
    idprofile: str,
    db: Session = Depends(get_db),
):
    scan = (
        db.query(Scan)
        .filter(
            Scan.idprofile == idprofile
        )
        .first()
    )

    if scan is None:
        raise HTTPException(
            status_code=404,
            detail="Profil introuvable",
        )

    return {
        "success": True,
        "id": scan.id,
        "idprofile": scan.idprofile,
        "discussion_data": scan.discussion_data,
        "created_at": scan.created_at,
        "updated_at": scan.updated_at,
    }


# ============================================================
# EXISTENCE D'UN PROFIL
#
# Cette route est préférable pour scan.dart.
# Elle ne renvoie PAS les réponses privées du profil.
# ============================================================

@app.get("/api/v1/scan/{idprofile}/exists")
def profile_exists(
    idprofile: str,
    db: Session = Depends(get_db),
):
    scan = (
        db.query(Scan.id)
        .filter(
            Scan.idprofile == idprofile
        )
        .first()
    )

    return {
        "success": True,
        "idprofile": idprofile,
        "exists": scan is not None,
    }


# ============================================================
# UPDATE SCAN
# ============================================================

@app.put("/api/v1/scan/{idprofile}")
def update_scan(
    idprofile: str,
    data: ScanUpdate,
    db: Session = Depends(get_db),
):
    scan = (
        db.query(Scan)
        .filter(
            Scan.idprofile == idprofile
        )
        .first()
    )

    if scan is None:
        raise HTTPException(
            status_code=404,
            detail="Profil introuvable",
        )

    scan.discussion_data = (
        data.discussion_data
    )

    db.commit()
    db.refresh(scan)

    return {
        "success": True,
        "message": "Profil mis à jour",
        "id": scan.id,
        "idprofile": scan.idprofile,
        "discussion_data":
            scan.discussion_data,
        "updated_at":
            scan.updated_at,
    }


# ============================================================
# DELETE SCAN
# ============================================================

@app.delete("/api/v1/scan/{idprofile}")
def delete_scan(
    idprofile: str,
    db: Session = Depends(get_db),
):
    scan = (
        db.query(Scan)
        .filter(
            Scan.idprofile == idprofile
        )
        .first()
    )

    if scan is None:
        raise HTTPException(
            status_code=404,
            detail="Profil introuvable",
        )

    db.delete(scan)
    db.commit()

    return {
        "success": True,
        "message": "Profil supprimé",
        "idprofile": idprofile,
    }


# ============================================================
# VERIFICATION DE DEUX PROFILS
#
# Permet à Flutter de vérifier les deux profils
# sans télécharger discussion_data.
# ============================================================

@app.post("/api/v1/match/verify")
def verify_match(
    data: MatchRequest,
    db: Session = Depends(get_db),
):
    my_profile_id = (
        data.my_profile_id.strip()
    )

    scanned_profile_id = (
        data.scanned_profile_id.strip()
    )

    if (
        not my_profile_id
        or not scanned_profile_id
    ):
        raise HTTPException(
            status_code=400,
            detail="Identifiant de profil invalide",
        )

    if (
        my_profile_id
        == scanned_profile_id
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Les deux profils doivent "
                "être différents"
            ),
        )

    my_exists = (
        db.query(Scan.id)
        .filter(
            Scan.idprofile
            == my_profile_id
        )
        .first()
        is not None
    )

    scanned_exists = (
        db.query(Scan.id)
        .filter(
            Scan.idprofile
            == scanned_profile_id
        )
        .first()
        is not None
    )

    return {
        "success":
            my_exists
            and scanned_exists,

        "ready":
            my_exists
            and scanned_exists,

        "my_profile_exists":
            my_exists,

        "scanned_profile_exists":
            scanned_exists,
    }


# ============================================================
# ANALYSE DE COMPATIBILITE AVEC GROQ
# ============================================================

@app.post("/api/v1/match")
def analyze_match(
    data: MatchRequest,
    db: Session = Depends(get_db),
):
    # ========================================================
    # VERIFIER GROQ
    # ========================================================

    if groq_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Le service d'intelligence "
                "artificielle n'est pas configuré"
            ),
        )

    # ========================================================
    # NETTOYER LES IDENTIFIANTS
    # ========================================================

    my_profile_id = (
        data.my_profile_id.strip()
    )

    scanned_profile_id = (
        data.scanned_profile_id.strip()
    )

    # ========================================================
    # VERIFICATIONS
    # ========================================================

    if (
        not my_profile_id
        or not scanned_profile_id
    ):
        raise HTTPException(
            status_code=400,
            detail="Identifiant de profil invalide",
        )

    if (
        my_profile_id
        == scanned_profile_id
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Impossible de comparer "
                "le même profil"
            ),
        )

    # ========================================================
    # RECUPERER LE PROFIL DU TELEPHONE QUI SCANNE
    # ========================================================

    my_profile = (
        db.query(Scan)
        .filter(
            Scan.idprofile
            == my_profile_id
        )
        .first()
    )

    if my_profile is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Le profil de l'utilisateur "
                "est introuvable"
            ),
        )

    # ========================================================
    # RECUPERER LE PROFIL DU QR CODE
    # ========================================================

    scanned_profile = (
        db.query(Scan)
        .filter(
            Scan.idprofile
            == scanned_profile_id
        )
        .first()
    )

    if scanned_profile is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Le profil scanné "
                "est introuvable"
            ),
        )

    # ========================================================
    # LANGUE
    # ========================================================

    languages = {
        "fr": "français",
        "en": "English",
        "es": "español",
        "pt": "português",
    }

    requested_language = (
        data.language
        .strip()
        .lower()
    )

    if (
        requested_language
        not in languages
    ):
        requested_language = "fr"

    language_name = languages[
        requested_language
    ]

    # ========================================================
    # DONNEES ENVOYEES A GROQ
    #
    # Les identifiants DMP ne sont pas nécessaires à l'IA.
    # Nous envoyons uniquement les réponses.
    # ========================================================

    profiles_payload = {
        "profile_a":
            my_profile.discussion_data,

        "profile_b":
            scanned_profile.discussion_data,
    }

    # ========================================================
    # INSTRUCTIONS POUR GROQ
    # ========================================================

    system_prompt = f"""
You are the compatibility analysis engine for the
Deep Matching application.

You receive two relationship profiles generated from
the Cupid questionnaire.

Compare the two profiles carefully.

IMPORTANT RULES:

1. Base the analysis ONLY on the information contained
   in the two profiles.

2. Never invent missing information.

3. Do not make medical or psychological diagnoses.

4. Do not judge either person morally.

5. A difference does not automatically mean
   incompatibility.

6. Identify agreements, differences and important
   subjects that the two people should discuss.

7. Give particular attention to:
   - relationship model
   - fidelity
   - marriage
   - children
   - religion
   - family
   - money and financial expectations
   - professional projects
   - country of residence and expatriation
   - household responsibilities
   - gender roles
   - communication
   - conflict management
   - lifestyle
   - alcohol and tobacco
   - education
   - social background
   - intimacy
   - consent
   - independence
   - red flags
   - long-term expectations

8. Treat sensitive personal information respectfully.

9. Do not claim that the relationship will succeed
   or fail.

10. Explain uncertainty when the available answers
    are insufficient.

11. Write ALL human-readable text in:
    {language_name}

Return ONLY a valid JSON object.

The JSON must use exactly this structure:

{{
  "summary": "General compatibility summary",

  "compatibilities": [
    "Compatible point 1",
    "Compatible point 2"
  ],

  "differences": [
    "Important difference 1",
    "Important difference 2"
  ],

  "attention_points": [
    "Topic that should be discussed 1",
    "Topic that should be discussed 2"
  ],

  "recommendation":
    "Neutral and nuanced conclusion"
}}

Do not add Markdown.
Do not add text before the JSON.
Do not add text after the JSON.
"""

    # ========================================================
    # MESSAGE UTILISATEUR
    # ========================================================

    user_prompt = (
        "Compare the following two Deep Matching "
        "profiles.\n\n"
        + json.dumps(
            profiles_payload,
            ensure_ascii=False,
        )
    )

    # ========================================================
    # APPEL GROQ
    # ========================================================

    try:
        completion = (
            groq_client
            .chat
            .completions
            .create(
                model=(
                    "llama-3.3-70b-versatile"
                ),

                messages=[
                    {
                        "role": "system",
                        "content":
                            system_prompt,
                    },
                    {
                        "role": "user",
                        "content":
                            user_prompt,
                    },
                ],

                # JSON Object Mode
                response_format={
                    "type":
                        "json_object"
                },

                temperature=0.2,

                max_completion_tokens=2500,

                stream=False,
            )
        )

    except Exception as exc:
        print(
            "=========================================="
        )

        print(
            "ERREUR GROQ :",
            repr(exc),
        )

        print(
            "=========================================="
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Impossible d'effectuer "
                "l'analyse de compatibilité"
            ),
        )

    # ========================================================
    # RECUPERER LA REPONSE
    # ========================================================

    try:
        content = (
            completion
            .choices[0]
            .message
            .content
        )

        if not content:
            raise ValueError(
                "Réponse Groq vide"
            )

        result = json.loads(
            content
        )

    except (
        json.JSONDecodeError,
        ValueError,
        IndexError,
        AttributeError,
    ) as exc:
        print(
            "=========================================="
        )

        print(
            "REPONSE GROQ INVALIDE :",
            repr(exc),
        )

        print(
            "=========================================="
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "La réponse de l'IA "
                "est invalide"
            ),
        )

    # ========================================================
    # VERIFIER LA STRUCTURE DU RESULTAT
    # ========================================================

    required_fields = [
        "summary",
        "compatibilities",
        "differences",
        "attention_points",
        "recommendation",
    ]

    for field in required_fields:
        if field not in result:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Réponse IA incomplète : "
                    f"{field}"
                ),
            )

    # ========================================================
    # VERIFIER LES LISTES
    # ========================================================

    if not isinstance(
        result["compatibilities"],
        list,
    ):
        result["compatibilities"] = []

    if not isinstance(
        result["differences"],
        list,
    ):
        result["differences"] = []

    if not isinstance(
        result["attention_points"],
        list,
    ):
        result["attention_points"] = []

    # ========================================================
    # LOG SERVEUR
    #
    # Ne pas afficher discussion_data dans les logs.
    # ========================================================

    print(
        "=========================================="
    )

    print(
        "DEEP MATCHING : ANALYSE TERMINEE"
    )

    print(
        "PROFIL A :",
        my_profile_id,
    )

    print(
        "PROFIL B :",
        scanned_profile_id,
    )

    print(
        "LANGUE :",
        requested_language,
    )

    print(
        "=========================================="
    )

    # ========================================================
    # REPONSE A FLUTTER
    # ========================================================

    return {
        "success": True,

        "my_profile_id":
            my_profile_id,

        "scanned_profile_id":
            scanned_profile_id,

        "language":
            requested_language,

        "result":
            result,
    }