"""Minimal OAuth 2.1 authorization server for a single owner.

ChatGPT (and other MCP clients) register dynamically, send the owner to /login,
the owner types the access code (contents of /workspace/h3-mcp/token), and the
client receives short-lived access tokens + refresh tokens.
State is persisted to oauth_state.json (tokens stored only as SHA-256 hashes)."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import secrets
import time
from pathlib import Path

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

SCOPE = "studio"
ACCESS_TTL = 24 * 3600          # 1 day
REFRESH_TTL = 90 * 24 * 3600    # 90 days
CODE_TTL = 300                  # 5 minutes
PENDING_TTL = 600               # login page valid for 10 minutes


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class OwnerOAuthProvider:
    def __init__(self, owner_code: str, public_url: str, state_file: Path):
        self.owner_code = owner_code
        self.public_url = public_url.rstrip("/")
        self.state_file = state_file
        self.clients: dict[str, dict] = {}
        self.access: dict[str, dict] = {}    # sha256(token) -> record
        self.refresh: dict[str, dict] = {}   # sha256(token) -> record
        self.codes: dict[str, AuthorizationCode] = {}
        self.pending: dict[str, dict] = {}
        self._load()

    # ---------------- persistence ----------------
    def _load(self):
        try:
            data = json.loads(self.state_file.read_text())
            self.clients = data.get("clients", {})
            now = time.time()
            self.access = {k: v for k, v in data.get("access", {}).items() if v["expires_at"] > now}
            self.refresh = {k: v for k, v in data.get("refresh", {}).items() if v["expires_at"] > now}
        except Exception:
            pass

    def _save(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"clients": self.clients, "access": self.access,
                                   "refresh": self.refresh}))
        tmp.chmod(0o600)
        tmp.replace(self.state_file)

    # ---------------- clients ----------------
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        info = self.clients.get(client_id)
        return OAuthClientInformationFull.model_validate(info) if info else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.clients[client_info.client_id] = client_info.model_dump(mode="json")
        self._save()

    # ---------------- authorization ----------------
    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        req = secrets.token_urlsafe(24)
        self.pending[req] = {"client_id": client.client_id, "params": params,
                             "expires_at": time.time() + PENDING_TTL, "attempts": 0,
                             "client_name": client.client_name or client.client_id}
        return f"{self.public_url}/login?req={req}"

    async def load_authorization_code(self, client, authorization_code: str):
        code = self.codes.get(authorization_code)
        if code and code.client_id == client.client_id and code.expires_at > time.time():
            return code
        return None

    async def exchange_authorization_code(self, client, authorization_code: AuthorizationCode) -> OAuthToken:
        self.codes.pop(authorization_code.code, None)
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.resource)

    # ---------------- tokens ----------------
    def _issue(self, client_id: str, scopes: list[str], resource: str | None) -> OAuthToken:
        now = time.time()
        access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(40)
        self.access[_h(access)] = {"client_id": client_id, "scopes": scopes,
                                   "expires_at": now + ACCESS_TTL, "resource": resource}
        self.refresh[_h(refresh)] = {"client_id": client_id, "scopes": scopes,
                                     "expires_at": now + REFRESH_TTL, "resource": resource}
        self._save()
        return OAuthToken(access_token=access, token_type="Bearer", expires_in=ACCESS_TTL,
                          scope=" ".join(scopes), refresh_token=refresh)

    async def load_refresh_token(self, client, refresh_token: str):
        rec = self.refresh.get(_h(refresh_token))
        if rec and rec["client_id"] == client.client_id and rec["expires_at"] > time.time():
            return RefreshToken(token=refresh_token, client_id=rec["client_id"], scopes=rec["scopes"],
                                expires_at=int(rec["expires_at"]))
        return None

    async def exchange_refresh_token(self, client, refresh_token: RefreshToken, scopes: list[str]) -> OAuthToken:
        rec = self.refresh.pop(_h(refresh_token.token), None) or {}
        return self._issue(client.client_id, scopes or refresh_token.scopes, rec.get("resource"))

    async def load_access_token(self, token: str):
        # The static owner code also works as a bearer token (used by test_client / API calls).
        if hmac.compare_digest(token, self.owner_code):
            return AccessToken(token=token, client_id="owner", scopes=[SCOPE], expires_at=None)
        rec = self.access.get(_h(token))
        if rec and rec["expires_at"] > time.time():
            return AccessToken(token=token, client_id=rec["client_id"], scopes=rec["scopes"],
                               expires_at=int(rec["expires_at"]), resource=rec.get("resource"))
        return None

    async def revoke_token(self, token) -> None:
        key = _h(token.token)
        self.access.pop(key, None)
        self.refresh.pop(key, None)
        self._save()

    # ---------------- login page ----------------
    def _page(self, req: str, client_name: str, error: str = "") -> HTMLResponse:
        err = f'<p class="err">{html.escape(error)}</p>' if error else ""
        return HTMLResponse(f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Creative Studio – Sign in</title>
<style>body{{font-family:system-ui,sans-serif;background:#111;color:#eee;display:grid;place-items:center;min-height:100vh;margin:0}}
form{{background:#1d1d1f;padding:28px;border-radius:14px;width:min(380px,90vw)}}h1{{font-size:20px;margin:0 0 6px}}
p{{color:#aaa;font-size:14px}}input{{width:100%;box-sizing:border-box;padding:12px;border-radius:8px;border:1px solid #444;background:#000;color:#fff;font-size:15px}}
button{{margin-top:14px;width:100%;padding:12px;border:0;border-radius:8px;background:#4b6bff;color:#fff;font-size:15px;cursor:pointer}}
.err{{color:#ff7b7b}}</style></head><body><form method="post" action="/login">
<h1>Creative Studio</h1><p><b>{html.escape(client_name)}</b> ขอเข้าใช้งานเซิร์ฟเวอร์ของคุณ<br>
ใส่รหัสเข้าใช้งาน (ผลของคำสั่ง <code>cat /workspace/h3-mcp/token</code>) เพื่ออนุญาต</p>{err}
<input type="password" name="code" autocomplete="off" autofocus placeholder="รหัสเข้าใช้งาน">
<input type="hidden" name="req" value="{html.escape(req)}"><button type="submit">อนุญาต</button></form></body></html>""")

    async def login_get(self, request: Request):
        req = request.query_params.get("req", "")
        p = self.pending.get(req)
        if not p or p["expires_at"] < time.time():
            return HTMLResponse("<p>ลิงก์หมดอายุ กรุณาเริ่มเชื่อมต่อใหม่จากแอป</p>", status_code=400)
        return self._page(req, p["client_name"])

    async def login_post(self, request: Request):
        form = await request.form()
        req, code = str(form.get("req", "")), str(form.get("code", "")).strip()
        p = self.pending.get(req)
        if not p or p["expires_at"] < time.time():
            return HTMLResponse("<p>ลิงก์หมดอายุ กรุณาเริ่มเชื่อมต่อใหม่จากแอป</p>", status_code=400)
        if not hmac.compare_digest(code, self.owner_code):
            p["attempts"] += 1
            await asyncio.sleep(1.5)
            if p["attempts"] >= 5:
                self.pending.pop(req, None)
                return HTMLResponse("<p>ใส่รหัสผิดหลายครั้ง กรุณาเริ่มเชื่อมต่อใหม่</p>", status_code=403)
            return self._page(req, p["client_name"], "รหัสไม่ถูกต้อง")
        self.pending.pop(req, None)
        params: AuthorizationParams = p["params"]
        auth_code = secrets.token_urlsafe(32)
        self.codes[auth_code] = AuthorizationCode(
            code=auth_code, scopes=params.scopes or [SCOPE], expires_at=time.time() + CODE_TTL,
            client_id=p["client_id"], code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=getattr(params, "resource", None))
        return RedirectResponse(construct_redirect_uri(str(params.redirect_uri), code=auth_code,
                                                       state=params.state), status_code=302)
