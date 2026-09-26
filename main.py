import json
import os
import traceback
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Scan


# ============================================================
# DATABASE
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Deep Matching API",
    description="API serveur de l'application Deep Matching",
    version="3.1.1",
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

groq_client: Optional[Groq] = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class ScanCreate(BaseModel):
    idprofile: str = Field(..., min_length=1, max_length=100)
    discussion_data: Dict[str, Any]


class ScanUpdate(BaseModel):
    discussion_data: Dict[str, Any]


class MatchRequest(BaseModel):
    my_profile_id: str = Field(..., min_length=1, max_length=100)
    scanned_profile_id: str = Field(..., min_length=1, max_length=100)
    language: str = "fr"


# ============================================================
# UTILITIES
# ============================================================

SUPPORTED_LANGUAGES = {
    "fr": "français",
    "en": "anglais",
    "es": "espagnol",
    "pt": "portugais",
}


def normalize_language(language: str) -> str:
    language = (language or "fr").lower().strip()

    if language not in SUPPORTED_LANGUAGES:
        return "fr"

    return language


def get_profile_or_404(
    db: Session,
    profile_id: str,
) -> Scan:

    profile = (
        db.query(Scan)
        .filter(Scan.idprofile == profile_id)
        .first()
    )

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profil introuvable : {profile_id}",
        )

    return profile


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "deep-matching-api",
        "version": "3.1.1",
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
# CREATE PROFILE
# ============================================================

@app.post(
    "/api/v1/scan",
    status_code=status.HTTP_201_CREATED,
)
def create_scan(
    payload: ScanCreate,
    db: Session = Depends(get_db),
):

    existing = (
        db.query(Scan)
        .filter(Scan.idprofile == payload.idprofile)
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ce profil existe déjà",
        )

    scan = Scan(
        idprofile=payload.idprofile,
        discussion_data=payload.discussion_data,
    )

    db.add(scan)

    try:
        db.commit()
        db.refresh(scan)

    except Exception as e:

        db.rollback()

        print(
            f"[DATABASE][CREATE] "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible d'enregistrer le profil",
        )

    return {
        "success": True,
        "message": "Profil enregistré",
        "id": scan.id,
        "idprofile": scan.idprofile,
    }


# ============================================================
# PROFILE EXISTS
# ============================================================

@app.get("/api/v1/scan/{idprofile}/exists")
def profile_exists(
    idprofile: str,
    db: Session = Depends(get_db),
):

    profile = (
        db.query(Scan)
        .filter(Scan.idprofile == idprofile)
        .first()
    )

    return {
        "idprofile": idprofile,
        "exists": profile is not None,
    }


# ============================================================
# GET PROFILE
# ============================================================

@app.get("/api/v1/scan/{idprofile}")
def get_scan(
    idprofile: str,
    db: Session = Depends(get_db),
):

    scan = get_profile_or_404(
        db,
        idprofile,
    )

    return {
        "id": scan.id,
        "idprofile": scan.idprofile,
        "discussion_data": scan.discussion_data,
        "created_at": scan.created_at,
        "updated_at": scan.updated_at,
    }


# ============================================================
# UPDATE PROFILE
# ============================================================

@app.put("/api/v1/scan/{idprofile}")
def update_scan(
    idprofile: str,
    payload: ScanUpdate,
    db: Session = Depends(get_db),
):

    scan = get_profile_or_404(
        db,
        idprofile,
    )

    scan.discussion_data = payload.discussion_data

    try:

        db.commit()
        db.refresh(scan)

    except Exception as e:

        db.rollback()

        print(
            f"[DATABASE][UPDATE] "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de mettre à jour le profil",
        )

    return {
        "success": True,
        "message": "Profil mis à jour",
        "idprofile": scan.idprofile,
    }


# ============================================================
# DELETE PROFILE
# ============================================================

@app.delete("/api/v1/scan/{idprofile}")
def delete_scan(
    idprofile: str,
    db: Session = Depends(get_db),
):

    scan = get_profile_or_404(
        db,
        idprofile,
    )

    try:

        db.delete(scan)
        db.commit()

    except Exception as e:

        db.rollback()

        print(
            f"[DATABASE][DELETE] "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossible de supprimer le profil",
        )

    return {
        "success": True,
        "message": "Profil supprimé",
        "idprofile": idprofile,
    }


# ============================================================
# VERIFY MATCH
# ============================================================

@app.post("/api/v1/match/verify")
def verify_match(
    payload: MatchRequest,
    db: Session = Depends(get_db),
):

    my_profile = (
        db.query(Scan)
        .filter(
            Scan.idprofile == payload.my_profile_id
        )
        .first()
    )

    scanned_profile = (
        db.query(Scan)
        .filter(
            Scan.idprofile == payload.scanned_profile_id
        )
        .first()
    )

    return {
        "ready": (
            my_profile is not None
            and scanned_profile is not None
        ),
        "my_profile_exists": my_profile is not None,
        "scanned_profile_exists": scanned_profile is not None,
    }


# ============================================================
# MATCH WITH GROQ
# ============================================================

@app.post("/api/v1/match")
def analyze_match(
    payload: MatchRequest,
    db: Session = Depends(get_db),
):

    # --------------------------------------------------------
    # Vérification Groq
    # --------------------------------------------------------

    if not GROQ_API_KEY or groq_client is None:

        print(
            "[GROQ] GROQ_API_KEY absente.",
            flush=True,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service d'analyse IA non configuré",
        )

    # --------------------------------------------------------
    # Langue
    # --------------------------------------------------------

    language = normalize_language(
        payload.language
    )

    language_name = SUPPORTED_LANGUAGES[
        language
    ]

    # --------------------------------------------------------
    # Charger les deux profils
    # --------------------------------------------------------

    my_profile = get_profile_or_404(
        db,
        payload.my_profile_id,
    )

    scanned_profile = get_profile_or_404(
        db,
        payload.scanned_profile_id,
    )

    print(
        "[MATCH] Analyse demandée : "
        f"{payload.my_profile_id} <-> "
        f"{payload.scanned_profile_id}",
        flush=True,
    )

    # --------------------------------------------------------
    # Préparation des données
    # --------------------------------------------------------

    profile_a_data = (
        my_profile.discussion_data or {}
    )

    profile_b_data = (
        scanned_profile.discussion_data or {}
    )

    # --------------------------------------------------------
    # Prompt système
    # --------------------------------------------------------

    system_prompt = f"""
Tu es Cupid, le moteur d'analyse de compatibilité
relationnelle de l'application Deep Matching.

Tu dois comparer deux profils à partir des réponses
fournies par les utilisateurs.

Ton rôle n'est pas de décider si deux personnes doivent
ou non commencer ou poursuivre une relation.

Tu dois identifier de manière neutre :

- leurs principaux points communs ;
- leurs différences importantes ;
- les sujets qui méritent une discussion ;
- les éventuels points d'attention ;
- une synthèse générale de leur compatibilité.

Ne donne aucun diagnostic médical ou psychologique.

N'invente aucune information qui n'est pas présente
dans les profils.

Si certaines informations sont insuffisantes,
indique-le clairement.

Réponds exclusivement en {language_name}.

IMPORTANT :

La réponse doit être exclusivement un objet JSON valide.

N'utilise pas de Markdown.
N'utilise pas ```json.
N'ajoute aucun texte avant ou après le JSON.

Structure obligatoire :

{{
    "summary": "texte",
    "compatibilities": [
        "élément 1",
        "élément 2"
    ],
    "differences": [
        "élément 1",
        "élément 2"
    ],
    "attention_points": [
        "élément 1",
        "élément 2"
    ],
    "recommendation": "texte"
}}
"""

    # --------------------------------------------------------
    # Message utilisateur
    # --------------------------------------------------------

    user_prompt = f"""
Analyse la compatibilité entre les deux profils suivants.

PROFIL DE L'UTILISATEUR :

{json.dumps(
    profile_a_data,
    ensure_ascii=False,
    indent=2,
    default=str,
)}

PROFIL SCANNÉ :

{json.dumps(
    profile_b_data,
    ensure_ascii=False,
    indent=2,
    default=str,
)}

Compare uniquement les informations disponibles.
"""

    # --------------------------------------------------------
    # Appel Groq
    # --------------------------------------------------------

    try:

        print(
            "[GROQ] Envoi de la requête...",
            flush=True,
        )

        completion = (
            groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.3,
                response_format={
                    "type": "json_object"
                },
            )
        )

        print(
            "[GROQ] Réponse reçue.",
            flush=True,
        )

        # ----------------------------------------------------
        # Récupération du contenu
        # ----------------------------------------------------

        if not completion.choices:

            raise ValueError(
                "Groq n'a retourné aucun choix."
            )

        content = (
            completion
            .choices[0]
            .message
            .content
        )

        if not content:

            raise ValueError(
                "Groq a retourné une réponse vide."
            )

        print(
            "[GROQ] Réponse reçue et non vide.",
            flush=True,
        )

        # ----------------------------------------------------
        # Conversion JSON
        # ----------------------------------------------------

        try:

            analysis = json.loads(content)

        except json.JSONDecodeError as json_error:

            print(
                "[GROQ][JSON] Réponse invalide : "
                f"{content[:1000]}",
                flush=True,
            )

            raise ValueError(
                "La réponse Groq n'est pas un JSON valide."
            ) from json_error

        # ----------------------------------------------------
        # Vérification minimale du résultat
        # ----------------------------------------------------

        required_fields = [
            "summary",
            "compatibilities",
            "differences",
            "attention_points",
            "recommendation",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in analysis
        ]

        if missing_fields:

            print(
                "[GROQ][JSON] Champs manquants : "
                f"{missing_fields}",
                flush=True,
            )

            raise ValueError(
                "Réponse Groq incomplète."
            )

        # ----------------------------------------------------
        # Succès
        # ----------------------------------------------------

        print(
            "[MATCH] Analyse terminée avec succès.",
            flush=True,
        )

        return {
            "success": True,
            "my_profile_id":
                payload.my_profile_id,
            "scanned_profile_id":
                payload.scanned_profile_id,
            "language": language,
            "analysis": analysis,
        }

    # --------------------------------------------------------
    # ERREUR GROQ
    # --------------------------------------------------------

    except Exception as e:

        print(
            "====================================",
            flush=True,
        )

        print(
            "[GROQ ERROR]",
            flush=True,
        )

        print(
            f"Type : {type(e).__name__}",
            flush=True,
        )

        print(
            f"Message : {e}",
            flush=True,
        )

        traceback.print_exc()

        print(
            "====================================",
            flush=True,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Impossible d'effectuer "
                "l'analyse de compatibilité"
            ),
        )