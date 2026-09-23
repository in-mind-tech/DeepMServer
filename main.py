from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Dict, Any

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
    description="API Backend de l'application Deep Matching - Cupid V3",
    version="3.0.0"
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
# SCHÉMAS PYDANTIC
# ============================================================

class ScanCreate(BaseModel):
    idprofile: str
    discussion_data: Dict[str, Any]


class ScanUpdate(BaseModel):
    discussion_data: Dict[str, Any]


# ============================================================
# ACCUEIL API
# ============================================================

@app.get("/")
def home():
    return {
        "application": "Deep Matching",
        "service": "Cupid V3 API",
        "version": "3.0.0",
        "status": "online"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "database": "PostgreSQL",
        "service": "deep-matching-api"
    }


# ============================================================
# INSERT
# POST /api/v1/scan
# ============================================================

@app.post(
    "/api/v1/scan",
    status_code=status.HTTP_201_CREATED
)
def create_scan(
    data: ScanCreate,
    db: Session = Depends(get_db)
):

    # Vérifier si le profil existe déjà
    existing_scan = (
        db.query(Scan)
        .filter(Scan.idprofile == data.idprofile)
        .first()
    )

    if existing_scan:
        raise HTTPException(
            status_code=409,
            detail="Ce profil existe déjà"
        )

    scan = Scan(
        idprofile=data.idprofile,
        discussion_data=data.discussion_data
    )

    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {
        "success": True,
        "message": "Profil enregistré",
        "id": scan.id,
        "idprofile": scan.idprofile
    }


# ============================================================
# SELECT
# GET /api/v1/scan/{idprofile}
# ============================================================

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
# UPDATE
# PUT /api/v1/scan/{idprofile}
# ============================================================

@app.put("/api/v1/scan/{idprofile}")
def update_scan(
    idprofile: str,
    data: ScanUpdate,
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

    scan.discussion_data = data.discussion_data

    db.commit()
    db.refresh(scan)

    return {
        "success": True,
        "message": "Profil mis à jour",
        "id": scan.id,
        "idprofile": scan.idprofile,
        "discussion_data": scan.discussion_data,
        "updated_at": scan.updated_at
    }


# ============================================================
# DELETE
# DELETE /api/v1/scan/{idprofile}
# ============================================================

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