import os
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from database import engine
from models import Scan, MatchResult, Historique


# ============================================================
# AUTHENTIFICATION SQLADMIN
# ============================================================
class AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")

        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD")

        if not admin_password:
            print(
                "[ADMIN AUTH] ADMIN_PASSWORD non configuré. "
                "Connexion refusée.",
                flush=True,
            )
            return False

        if username == admin_username and password == admin_password:
            request.session.update({"admin_authenticated": True})
            print(
                f"[ADMIN AUTH] Connexion réussie pour {username}",
                flush=True,
            )
            return True

        print(
            f"[ADMIN AUTH] Échec de connexion pour {username}",
            flush=True,
        )
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get("admin_authenticated", False)


# ============================================================
# VUES ADMIN
# ============================================================
class ScanAdmin(ModelView, model=Scan):
    name = "Profil (Scan)"
    name_plural = "Profils (Scan)"
    icon = "fa-solid fa-user"
    column_list = [
        Scan.id,
        Scan.idprofile,
        Scan.created_at,
        Scan.updated_at,
    ]
    column_readonly = [
        Scan.id,
        Scan.created_at,
        Scan.updated_at,
    ]


class MatchResultAdmin(ModelView, model=MatchResult):
    name = "Résultat de Match"
    name_plural = "Résultats de Match"
    icon = "fa-solid fa-heart"
    column_list = [
        MatchResult.id,
        MatchResult.match_id,
        MatchResult.profile_a_id,
        MatchResult.profile_b_id,
        MatchResult.match_a_to_b,
        MatchResult.match_b_to_a,
        MatchResult.scanner_received,
        MatchResult.owner_received,
        MatchResult.created_at,
    ]
    column_readonly = [
        MatchResult.id,
        MatchResult.match_id,
        MatchResult.created_at,
    ]


class HistoriqueAdmin(ModelView, model=Historique):
    name = "Historique (Admin)"
    name_plural = "Historique (Admin)"
    icon = "fa-solid fa-chart-bar"
    column_list = [
        Historique.id,
        Historique.my_gender,
        Historique.scanned_gender,
        Historique.my_country,
        Historique.scanned_country,
        Historique.match_my_to_their,
        Historique.match_their_to_my,
        Historique.datescan,
    ]
    column_readonly = [
        Historique.id,
        Historique.datescan,
    ]
    can_create = False
    can_edit = False
    can_delete = True


# ============================================================
# CONFIGURATION DE L'ADMIN
# ============================================================
def setup_admin(app):
    secret_key = os.getenv("SECRET_KEY")

    if not secret_key:
        print(
            "[ADMIN] ATTENTION : SECRET_KEY non configurée. "
            "Utilisation d'une clé par défaut (DANGEREUX en production).",
            flush=True,
        )
        secret_key = "change-me-in-production-immediately"

    print(
        f"[ADMIN] Configuration SQLAdmin sur /admin "
        f"(username={os.getenv('ADMIN_USERNAME', 'admin')})",
        flush=True,
    )

    admin = Admin(
        app,
        engine,
        title="Deep Matching Administration",
        base_url="/admin",
        authentication_backend=AdminAuth(secret_key=secret_key),
    )

    admin.add_view(ScanAdmin)
    admin.add_view(MatchResultAdmin)
    admin.add_view(HistoriqueAdmin)

    return admin