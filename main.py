from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Any, Dict

from database import Base, engine, get_db
from models import Scan


# ============================================================
# CRÉATION DES TABLES
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="Deep Matching API",
    description="Backend Deep Matching - Cupid V3",
    version="3.0.0"
)


# ============================================================
# MODÈLES PYDANTIC
# ============================================================

# Utilisé lors de la création d'un profil
class ScanCreate(BaseModel):
    idprofile: str
    discussion_data: Dict[str, Any]


# Utilisé lors de la modification d'un profil
class ScanUpdate(BaseModel):
    discussion_data: Dict[str, Any]


# ============================================================
# ROUTE PRINCIPALE
# ============================================================

@app.get("/")
def root():
    return {
        "application": "Deep Matching",
        "version": "Cupid V3",
        "status": "online"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# POST /api/v1/scan
# Création d'un nouveau profil
# ============================================================

@app.post("/api/v1/scan", status_code=201)
def create_scan(
    data: ScanCreate,
    db: Session = Depends(get_db)
):

    # --------------------------------------------------------
    # Vérifier si le profil existe déjà
    # --------------------------------------------------------

    existing_scan = (
        db.query(Scan)
        .filter(Scan.idprofile == data.idprofile)
        .first()
    )

    if existing_scan:
        raise HTTPException(
            status_code=409,
            detail="Ce profil existe déjà sur le serveur."
        )

    # --------------------------------------------------------
    # Création du profil
    # --------------------------------------------------------

    new_scan = Scan(
        idprofile=data.idprofile,
        discussion_data=data.discussion_data
    )

    db.add(new_scan)

    try:
        db.commit()
        db.refresh(new_scan)

    except Exception as e:

        db.rollback()

        print("ERREUR DATABASE POST :", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Erreur database : {str(e)}"
        )

    # --------------------------------------------------------
    # Réponse
    # --------------------------------------------------------

    return {
        "success": True,
        "message": "Profil enregistré avec succès.",
        "data": {
            "id": new_scan.id,
            "idprofile": new_scan.idprofile,
            "discussion_data": new_scan.discussion_data,
            "created_at": new_scan.created_at
        }
    }

@app.delete("/api/v1/scan/{idprofile}")
def delete_scan(
    idprofile: str,
    db: Session = Depends(get_db)
):
    scan = (
        db.query(Scan)
        .filter(Scan.idprofile == idprofile)
        .first()
    )

    if scan is None:
        raise HTTPException(
            status_code=404,
            detail="Profil introuvable"
        )

    db.delete(scan)
    db.commit()

    return {
        "success": True,
        "message": "Profil supprimé",
        "idprofile": idprofile
    }



@app.get("/api/v1/scan/{idprofile}")
def get_scan(
    idprofile: str,
    db: Session = Depends(get_db)
):
    scan = (
        db.query(Scan)
        .filter(Scan.idprofile == idprofile)
        .first()
    )

    if scan is None:
        raise HTTPException(
            status_code=404,
            detail="Profil introuvable"
        )

    return {
        "success": True,
        "id": scan.id,
        "idprofile": scan.idprofile,
        "discussion_data": scan.discussion_data,
        "created_at": scan.created_at,
        "updated_at": scan.updated_at
    }




# ============================================================
# PUT /api/v1/scan/{idprofile}
# Modification d'un profil existant
# ============================================================

@app.put("/api/v1/scan/{idprofile}")
def update_scan(
    idprofile: str,
    data: ScanUpdate,
    db: Session = Depends(get_db)
):

    # --------------------------------------------------------
    # Rechercher le profil
    # --------------------------------------------------------

    scan = (
        db.query(Scan)
        .filter(Scan.idprofile == idprofile)
        .first()
    )

    # --------------------------------------------------------
    # Profil inexistant
    # --------------------------------------------------------

    if not scan:
        raise HTTPException(
            status_code=404,
            detail="Profil introuvable."
        )

    # --------------------------------------------------------
    # Modification de discussion_data
    # --------------------------------------------------------

    scan.discussion_data = data.discussion_data

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    try:
        db.commit()
        db.refresh(scan)

    except Exception as e:

        db.rollback()

        print("ERREUR DATABASE PUT :", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Erreur database : {str(e)}"
        )

    # --------------------------------------------------------
    # Réponse
    # --------------------------------------------------------

    return {
        "success": True,
        "message": "Profil mis à jour avec succès.",
        "data": {
            "id": scan.id,
            "idprofile": scan.idprofile,
            "discussion_data": scan.discussion_data,
            "created_at": scan.created_at,
            "updated_at": scan.updated_at
        }
    }