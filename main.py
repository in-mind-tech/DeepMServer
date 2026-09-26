import json
import os
import traceback
import uuid

from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    status,
)

from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Scan, MatchResult, Historique


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
    version="3.4.0",
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

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

groq_client: Optional[Groq] = None

if GROQ_API_KEY:
    groq_client = Groq(
        api_key=GROQ_API_KEY
    )


# ============================================================
# PYDANTIC
# ============================================================

class ScanCreate(BaseModel):
    idprofile: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    discussion_data: Dict[str, Any]


class ScanUpdate(BaseModel):
    discussion_data: Dict[str, Any]


class MatchRequest(BaseModel):
    my_profile_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    scanned_profile_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    language: str = "fr"


# ============================================================
# LANGUES
# ============================================================

SUPPORTED_LANGUAGES = {
    "fr": "français",
    "en": "English",
    "es": "español",
    "pt": "português",
}


# ============================================================
# CATÉGORIES INTERNES DE TENSION
#
# Ces clés sont utilisées par l'IA et PostgreSQL.
# Elles ne sont jamais envoyées telles quelles à l'utilisateur.
# ============================================================

ALLOWED_TENSIONS = {
    "sexuality",
    "personality",
    "religion",
    "conflict_management",
    "children",
    "fidelity",
    "relationship_model",
    "marriage",
    "finances",
    "family",
    "lifestyle",
    "independence",
    "communication",
    "values",
    "future_projects",
    "other",
}


# ============================================================
# TRADUCTION DES TENSIONS
# ============================================================

TENSION_LABELS = {

    "fr": {
        "sexuality": "Sexualité",
        "personality": "Personnalité et comportement",
        "religion": "Religion et convictions",
        "conflict_management": "Gestion des conflits",
        "children": "Projet parental et enfants",
        "fidelity": "Fidélité et confiance",
        "relationship_model": "Vision de la relation",
        "marriage": "Vision du mariage",
        "finances": "Gestion financière",
        "family": "Famille et entourage",
        "lifestyle": "Mode de vie",
        "independence": "Indépendance et espace personnel",
        "communication": "Communication dans le couple",
        "values": "Valeurs et convictions",
        "future_projects": "Projets de vie",
        "other": "Autres différences importantes",
    },

    "en": {
        "sexuality": "Sexuality",
        "personality": "Personality and behaviour",
        "religion": "Religion and beliefs",
        "conflict_management": "Conflict management",
        "children": "Parenthood and children",
        "fidelity": "Fidelity and trust",
        "relationship_model": "Relationship expectations",
        "marriage": "Views on marriage",
        "finances": "Financial management",
        "family": "Family and social environment",
        "lifestyle": "Lifestyle",
        "independence": "Independence and personal space",
        "communication": "Communication in the relationship",
        "values": "Values and beliefs",
        "future_projects": "Life plans",
        "other": "Other important differences",
    },

    "es": {
        "sexuality": "Sexualidad",
        "personality": "Personalidad y comportamiento",
        "religion": "Religión y convicciones",
        "conflict_management": "Gestión de conflictos",
        "children": "Proyecto parental e hijos",
        "fidelity": "Fidelidad y confianza",
        "relationship_model": "Visión de la relación",
        "marriage": "Visión del matrimonio",
        "finances": "Gestión financiera",
        "family": "Familia y entorno",
        "lifestyle": "Estilo de vida",
        "independence": "Independencia y espacio personal",
        "communication": "Comunicación en la pareja",
        "values": "Valores y convicciones",
        "future_projects": "Proyectos de vida",
        "other": "Otras diferencias importantes",
    },

    "pt": {
        "sexuality": "Sexualidade",
        "personality": "Personalidade e comportamento",
        "religion": "Religião e convicções",
        "conflict_management": "Gestão de conflitos",
        "children": "Projeto parental e filhos",
        "fidelity": "Fidelidade e confiança",
        "relationship_model": "Visão do relacionamento",
        "marriage": "Visão do casamento",
        "finances": "Gestão financeira",
        "family": "Família e convívio social",
        "lifestyle": "Estilo de vida",
        "independence": "Independência e espaço pessoal",
        "communication": "Comunicação no relacionamento",
        "values": "Valores e convicções",
        "future_projects": "Projetos de vida",
        "other": "Outras diferenças importantes",
    },
}


# ============================================================
# UTILITAIRES
# ============================================================

def normalize_language(
    language: str,
) -> str:

    value = (
        language or "fr"
    ).strip().lower()

    if value not in SUPPORTED_LANGUAGES:
        return "fr"

    return value


def translate_tensions(
    tensions: List[str],
    language: str,
) -> List[str]:

    language = normalize_language(
        language
    )

    labels = TENSION_LABELS.get(
        language,
        TENSION_LABELS["fr"],
    )

    translated: List[str] = []

    for tension in tensions:

        label = labels.get(
            tension
        )

        if (
            label
            and label not in translated
        ):
            translated.append(
                label
            )

    return translated


def clamp_score(
    value: Any,
) -> int:

    try:
        score = int(
            round(
                float(value)
            )
        )

    except (
        ValueError,
        TypeError,
    ):
        return 0

    return max(
        0,
        min(
            100,
            score,
        ),
    )


def get_profile_or_404(
    db: Session,
    profile_id: str,
) -> Scan:

    profile = (
        db.query(Scan)
        .filter(
            Scan.idprofile
            == profile_id
        )
        .first()
    )

    if profile is None:

        raise HTTPException(
            status_code=404,
            detail="Profil introuvable",
        )

    return profile


def safe_text(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    return text[:100]


def first_value(
    data: Dict[str, Any],
    keys: List[str],
) -> Optional[str]:

    for key in keys:

        if key in data:

            value = safe_text(
                data.get(key)
            )

            if value:
                return value

    return None


def extract_general_information(
    data: Dict[str, Any],
) -> Dict[str, Optional[str]]:

    return {

        "gender": first_value(
            data,
            [
                "gender",
                "sex",
                "my_gender",
            ],
        ),

        "orientation": first_value(
            data,
            [
                "orientation",
                "sexual_orientation",
                "my_orientation",
                "relationship_orientation",
            ],
        ),

        "country": first_value(
            data,
            [
                "country",
                "my_country",
                "country_of_residence",
                "residence_country",
            ],
        ),
    }


# ============================================================
# CONSTRUCTION DE LA RÉPONSE
# ============================================================

def build_viewer_result(
    result: MatchResult,
    viewer_profile_id: str,
    language: str,
) -> Dict[str, Any]:

    language = normalize_language(
        language
    )

    translated_tensions = translate_tensions(
        result.tensions or [],
        language,
    )

    # --------------------------------------------------------
    # PROFIL A = utilisateur qui a effectué le scan
    # --------------------------------------------------------

    if (
        viewer_profile_id
        == result.profile_a_id
    ):

        return {
            "success": True,
            "match_id": result.match_id,
            "language": language,

            "compatibility": {
                "my_profile_to_their_expectations":
                    result.match_a_to_b,

                "their_profile_to_my_expectations":
                    result.match_b_to_a,
            },

            "tensions":
                translated_tensions,
        }

    # --------------------------------------------------------
    # PROFIL B = propriétaire du QR
    #
    # Les pourcentages sont inversés.
    # --------------------------------------------------------

    if (
        viewer_profile_id
        == result.profile_b_id
    ):

        return {
            "success": True,
            "match_id": result.match_id,
            "language": language,

            "compatibility": {
                "my_profile_to_their_expectations":
                    result.match_b_to_a,

                "their_profile_to_my_expectations":
                    result.match_a_to_b,
            },

            "tensions":
                translated_tensions,
        }

    raise HTTPException(
        status_code=403,
        detail=(
            "Ce profil ne participe "
            "pas à ce match"
        ),
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service":
            "deep-matching-api",
        "version":
            "3.4.0",
        "groq_model":
            GROQ_MODEL,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "database": "PostgreSQL",
        "service":
            "deep-matching-api",
        "groq_configured":
            bool(GROQ_API_KEY),
        "version":
            "3.4.0",
    }


# ============================================================
# CREATE / SYNC PROFILE
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
        .filter(
            Scan.idprofile
            == payload.idprofile
        )
        .first()
    )

    # --------------------------------------------------------
    # Profil existant : mise à jour
    # --------------------------------------------------------

    if existing:

        existing.discussion_data = (
            payload.discussion_data
        )

        try:

            db.commit()
            db.refresh(
                existing
            )

        except Exception as e:

            db.rollback()

            print(
                "[DATABASE][SYNC] "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Impossible de "
                    "synchroniser le profil"
                ),
            )

        return {
            "success": True,
            "message":
                "Profil synchronisé",
            "id":
                existing.id,
            "idprofile":
                existing.idprofile,
        }

    # --------------------------------------------------------
    # Nouveau profil
    # --------------------------------------------------------

    scan = Scan(
        idprofile=payload.idprofile,
        discussion_data=(
            payload.discussion_data
        ),
    )

    db.add(
        scan
    )

    try:

        db.commit()
        db.refresh(
            scan
        )

    except Exception as e:

        db.rollback()

        print(
            "[DATABASE][CREATE] "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible d'enregistrer "
                "le profil"
            ),
        )

    return {
        "success": True,
        "message":
            "Profil enregistré",
        "id":
            scan.id,
        "idprofile":
            scan.idprofile,
    }


# ============================================================
# PROFILE EXISTS
# ============================================================

@app.get(
    "/api/v1/scan/{idprofile}/exists"
)
def profile_exists(
    idprofile: str,
    db: Session = Depends(get_db),
):

    profile = (
        db.query(Scan)
        .filter(
            Scan.idprofile
            == idprofile
        )
        .first()
    )

    return {
        "idprofile":
            idprofile,
        "exists":
            profile is not None,
    }


# ============================================================
# GET PROFILE
#
# ATTENTION :
# discussion_data contient les réponses privées Cupid.
# Cet endpoint devra être protégé ou supprimé avant production.
# ============================================================

@app.get(
    "/api/v1/scan/{idprofile}"
)
def get_scan(
    idprofile: str,
    db: Session = Depends(get_db),
):

    scan = get_profile_or_404(
        db,
        idprofile,
    )

    return {
        "id":
            scan.id,

        "idprofile":
            scan.idprofile,

        "discussion_data":
            scan.discussion_data,

        "created_at":
            scan.created_at,

        "updated_at":
            scan.updated_at,
    }


# ============================================================
# UPDATE PROFILE
# ============================================================

@app.put(
    "/api/v1/scan/{idprofile}"
)
def update_scan(
    idprofile: str,
    payload: ScanUpdate,
    db: Session = Depends(get_db),
):

    scan = get_profile_or_404(
        db,
        idprofile,
    )

    scan.discussion_data = (
        payload.discussion_data
    )

    try:

        db.commit()
        db.refresh(
            scan
        )

    except Exception as e:

        db.rollback()

        print(
            "[DATABASE][UPDATE] "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible de mettre "
                "à jour le profil"
            ),
        )

    return {
        "success": True,
        "message":
            "Profil mis à jour",
        "idprofile":
            scan.idprofile,
    }


# ============================================================
# DELETE PROFILE
# ============================================================

@app.delete(
    "/api/v1/scan/{idprofile}"
)
def delete_scan(
    idprofile: str,
    db: Session = Depends(get_db),
):

    scan = get_profile_or_404(
        db,
        idprofile,
    )

    try:

        db.delete(
            scan
        )

        db.commit()

    except Exception as e:

        db.rollback()

        print(
            "[DATABASE][DELETE] "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible de supprimer "
                "le profil"
            ),
        )

    return {
        "success": True,
        "message":
            "Profil supprimé",
        "idprofile":
            idprofile,
    }


# ============================================================
# VERIFY MATCH
# ============================================================

@app.post(
    "/api/v1/match/verify"
)
def verify_match(
    payload: MatchRequest,
    db: Session = Depends(get_db),
):

    my_profile = (
        db.query(Scan)
        .filter(
            Scan.idprofile
            == payload.my_profile_id
        )
        .first()
    )

    scanned_profile = (
        db.query(Scan)
        .filter(
            Scan.idprofile
            == payload.scanned_profile_id
        )
        .first()
    )

    return {
        "ready": (
            my_profile is not None
            and scanned_profile is not None
        ),

        "my_profile_exists":
            my_profile is not None,

        "scanned_profile_exists":
            scanned_profile is not None,
    }


# ============================================================
# PROMPT DEEP MATCHING
# ============================================================

def build_match_prompt(
    profile_a: Dict[str, Any],
    profile_b: Dict[str, Any],
) -> str:

    return f"""
Tu es le moteur d'analyse sémantique de Deep Matching.

============================================================
OBJECTIF
============================================================

Tu compares deux questionnaires relationnels.

Tu dois produire deux scores directionnels indépendants.

A_TO_B :

Dans quelle mesure ce que la personne A déclare être
correspond-il à ce que la personne B déclare rechercher
chez un partenaire ?

B_TO_A :

Dans quelle mesure ce que la personne B déclare être
correspond-il à ce que la personne A déclare rechercher
chez un partenaire ?

Les deux scores peuvent être différents.

============================================================
SIGNIFICATION DES SCORES
============================================================

Les scores sont compris entre 0 et 100.

Ils mesurent uniquement la correspondance entre les
caractéristiques déclarées d'une personne et les attentes
déclarées de l'autre.

Ils ne représentent PAS :

- une probabilité de réussite du couple ;
- une probabilité de mariage ;
- une probabilité de séparation ;
- un diagnostic psychologique ;
- un risque de violence ;
- un risque de suicide ;
- un risque de dépression.

============================================================
IMPORTANCE DES CRITÈRES
============================================================

Toutes les réponses n'ont pas le même poids.

Une préférence légère doit avoir une influence limitée.

Une attente importante doit avoir une influence plus forte.

Une condition explicitement présentée comme :

- indispensable ;
- obligatoire ;
- non négociable ;
- rédhibitoire ;
- refus absolu ;
- impossible à accepter

doit avoir une influence très importante sur la direction
concernée.

Ne transforme toutefois pas une préférence ordinaire en
condition absolue.

============================================================
ORIENTATION ET TYPE DE PARTENAIRE
============================================================

Analyse uniquement ce qui est explicitement déclaré
concernant :

- le genre ;
- l'orientation ;
- le genre recherché ;
- le type de partenaire recherché.

Si les déclarations montrent clairement que A ne correspond
pas au type de partenaire recherché par B, cela doit fortement
réduire A_TO_B.

Si B ne correspond pas au type de partenaire recherché par A,
cela doit fortement réduire B_TO_A.

N'invente jamais une orientation, une préférence ou une
caractéristique absente du questionnaire.

============================================================
CRITÈRES IMPORTANTS
============================================================

Prends notamment en compte, lorsqu'ils sont effectivement
présents dans les réponses :

- sexualité ;
- religion et convictions ;
- enfants ;
- fidélité ;
- mariage ;
- modèle relationnel ;
- finances ;
- famille ;
- communication ;
- gestion des conflits ;
- indépendance ;
- personnalité et comportement ;
- mode de vie ;
- valeurs ;
- projets futurs.

Lorsqu'une attente clairement importante entre en conflit avec
la caractéristique déclarée de l'autre personne, elle doit
influencer principalement le score directionnel correspondant.

============================================================
TENSIONS
============================================================

Les tensions servent seulement à signaler les grands thèmes
sur lesquels les deux personnes pourraient souhaiter discuter.

Elles ne constituent pas une prédiction de conflit.

Retourne uniquement les clés suivantes :

sexuality
personality
religion
conflict_management
children
fidelity
relationship_model
marriage
finances
family
lifestyle
independence
communication
values
future_projects
other

N'ajoute une catégorie que lorsqu'une différence ou une
attente réellement pertinente est présente.

============================================================
CONFIDENTIALITÉ
============================================================

Les réponses Cupid sont privées.

Ne reproduis jamais une réponse exacte dans la sortie.

Ne donne aucune explication permettant à l'autre personne de
reconstituer une réponse privée.

La sortie ne doit contenir :

- aucun résumé ;
- aucune liste de points communs ;
- aucune liste détaillée de différences ;
- aucune citation ;
- aucune justification textuelle.

============================================================
FORMAT OBLIGATOIRE
============================================================

Retourne UNIQUEMENT un objet JSON valide exactement de cette
forme :

{{
    "a_to_b": 82,
    "b_to_a": 71,
    "tensions": [
        "religion",
        "conflict_management"
    ]
}}

a_to_b doit être un nombre entre 0 et 100.

b_to_a doit être un nombre entre 0 et 100.

tensions doit être une liste contenant uniquement les clés
autorisées.

Aucun Markdown.
Aucun commentaire.
Aucun texte avant le JSON.
Aucun texte après le JSON.

============================================================
PROFIL A
============================================================

{json.dumps(
    profile_a,
    ensure_ascii=False,
    indent=2,
    default=str,
)}

============================================================
PROFIL B
============================================================

{json.dumps(
    profile_b,
    ensure_ascii=False,
    indent=2,
    default=str,
)}
"""


# ============================================================
# ANALYSE MATCH
# ============================================================

@app.post(
    "/api/v1/match"
)
def analyze_match(
    payload: MatchRequest,
    db: Session = Depends(get_db),
):

    # --------------------------------------------------------
    # Vérification GROQ
    # --------------------------------------------------------

    if (
        not GROQ_API_KEY
        or groq_client is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Service d'analyse IA "
                "non configuré"
            ),
        )

    # --------------------------------------------------------
    # Empêcher l'auto-match
    # --------------------------------------------------------

    if (
        payload.my_profile_id
        == payload.scanned_profile_id
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Un profil ne peut pas "
                "être comparé à lui-même"
            ),
        )

    # --------------------------------------------------------
    # Récupération des profils
    # --------------------------------------------------------

    profile_a = get_profile_or_404(
        db,
        payload.my_profile_id,
    )

    profile_b = get_profile_or_404(
        db,
        payload.scanned_profile_id,
    )

    data_a = (
        profile_a.discussion_data
        or {}
    )

    data_b = (
        profile_b.discussion_data
        or {}
    )

    print(
        "[MATCH] "
        f"{payload.my_profile_id}"
        " <-> "
        f"{payload.scanned_profile_id}",
        flush=True,
    )

    # --------------------------------------------------------
    # Construction du prompt
    # --------------------------------------------------------

    prompt = build_match_prompt(
        data_a,
        data_b,
    )

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    try:

        print(
            "[GROQ] "
            f"Model: {GROQ_MODEL}",
            flush=True,
        )

        completion = (
            groq_client
            .chat
            .completions
            .create(
                model=GROQ_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyse les deux profils "
                            "relationnels. "
                            "Retourne uniquement "
                            "le JSON demandé."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                temperature=0.1,

                response_format={
                    "type": "json_object"
                },
            )
        )

        if not completion.choices:
            raise ValueError(
                "Aucun résultat Groq"
            )

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

        print(
            "[GROQ][RAW] "
            f"{content}",
            flush=True,
        )

        result = json.loads(
            content
        )

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "Format JSON Groq invalide"
            )

    except Exception as e:

        print(
            "================================",
            flush=True,
        )

        print(
            "[GROQ ERROR]",
            flush=True,
        )

        print(
            f"Type: {type(e).__name__}",
            flush=True,
        )

        print(
            f"Message: {e}",
            flush=True,
        )

        traceback.print_exc()

        print(
            "================================",
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Impossible d'effectuer "
                "l'analyse de compatibilité"
            ),
        )

    # ========================================================
    # SCORES
    # ========================================================

    match_a_to_b = clamp_score(
        result.get(
            "a_to_b",
            0,
        )
    )

    match_b_to_a = clamp_score(
        result.get(
            "b_to_a",
            0,
        )
    )

    # ========================================================
    # TENSIONS INTERNES
    # ========================================================

    raw_tensions = result.get(
        "tensions",
        [],
    )

    tensions: List[str] = []

    if isinstance(
        raw_tensions,
        list,
    ):

        for item in raw_tensions:

            tension = (
                str(item)
                .strip()
                .lower()
            )

            if (
                tension
                in ALLOWED_TENSIONS
                and tension
                not in tensions
            ):
                tensions.append(
                    tension
                )

    # ========================================================
    # MATCH ID
    # ========================================================

    match_id = (
        "DM-"
        + uuid.uuid4()
        .hex
        .upper()
    )

    # ========================================================
    # RÉSULTAT DU MATCH
    # ========================================================

    match_result = MatchResult(
        match_id=match_id,

        profile_a_id=(
            payload.my_profile_id
        ),

        profile_b_id=(
            payload.scanned_profile_id
        ),

        match_a_to_b=(
            match_a_to_b
        ),

        match_b_to_a=(
            match_b_to_a
        ),

        tensions=tensions,

        summary=None,
    )

    # ========================================================
    # INFORMATIONS POUR HISTORIQUE
    # ========================================================

    info_a = (
        extract_general_information(
            data_a
        )
    )

    info_b = (
        extract_general_information(
            data_b
        )
    )

    # ========================================================
    # HISTORIQUE
    # ========================================================

    historique = Historique(

        my_gender=(
            info_a["gender"]
        ),

        scanned_gender=(
            info_b["gender"]
        ),

        my_orientation=(
            info_a["orientation"]
        ),

        scanned_orientation=(
            info_b["orientation"]
        ),

        my_country=(
            info_a["country"]
        ),

        scanned_country=(
            info_b["country"]
        ),

        match_my_to_their=(
            match_a_to_b
        ),

        match_their_to_my=(
            match_b_to_a
        ),

        # On conserve uniquement les catégories,
        # jamais les réponses Cupid.
        conflit=tensions,
    )

    # ========================================================
    # ENREGISTREMENT
    # ========================================================

    try:

        db.add(
            match_result
        )

        db.add(
            historique
        )

        db.commit()

        db.refresh(
            match_result
        )

    except Exception as e:

        db.rollback()

        print(
            "[DATABASE][MATCH] "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Analyse réalisée mais "
                "impossible d'enregistrer "
                "le résultat"
            ),
        )

    # ========================================================
    # RÉPONSE POUR LE SCANNEUR
    # ========================================================

    response = build_viewer_result(
        match_result,
        payload.my_profile_id,
        payload.language,
    )

    print(
        "[MATCH][RESPONSE] "
        f"{response}",
        flush=True,
    )

    return response


# ============================================================
# DERNIER MATCH D'UN PROFIL
#
# IMPORTANT :
# Cette route ne se trouve plus sous :
#
# /api/v1/match/latest/...
#
# Cela évite la collision avec :
#
# /api/v1/match/{match_id}/{viewer_profile_id}
# ============================================================

@app.get(
    "/api/v1/profile/{viewer_profile_id}/latest-match"
)
def get_latest_match(
    viewer_profile_id: str,
    language: str = Query(
        default="fr",
    ),
    db: Session = Depends(get_db),
):

    result = (
        db.query(MatchResult)
        .filter(
            or_(
                MatchResult.profile_a_id
                == viewer_profile_id,

                MatchResult.profile_b_id
                == viewer_profile_id,
            )
        )
        .order_by(
            MatchResult.created_at.desc(),
            MatchResult.id.desc(),
        )
        .first()
    )

    if result is None:

        return {
            "success": True,
            "found": False,
        }

    viewer_result = build_viewer_result(
        result,
        viewer_profile_id,
        language,
    )

    viewer_result["found"] = True

    return viewer_result


# ============================================================
# RÉCUPÉRER UN MATCH PRÉCIS
#
# Exemple :
#
# /api/v1/match/DM-XXX/DMP-XXX?language=fr
# ============================================================

@app.get(
    "/api/v1/match/{match_id}/{viewer_profile_id}"
)
def get_match_result(
    match_id: str,
    viewer_profile_id: str,
    language: str = Query(
        default="fr",
    ),
    db: Session = Depends(get_db),
):

    result = (
        db.query(MatchResult)
        .filter(
            MatchResult.match_id
            == match_id
        )
        .first()
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail="Match introuvable",
        )

    return build_viewer_result(
        result,
        viewer_profile_id,
        language,
    )