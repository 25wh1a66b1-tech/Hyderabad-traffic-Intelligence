"""Session-state authentication and role permissions for the Streamlit dashboard."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import streamlit as st


ROLE_AUTHORITIES: Final = "Municipal & Infrastructure Authorities"
ROLE_FLEET: Final = "Commercial Fleet & Transit Operations"
ROLE_RESEARCH: Final = "Transportation Researchers & Developers"
ROLE_SECURITY: Final = "Traffic Monitoring & Security"
ROLES: Final = (ROLE_AUTHORITIES, ROLE_FLEET, ROLE_RESEARCH, ROLE_SECURITY)
AUTH_DB_PATH: Final = Path(__file__).resolve().parent.parent / "cache" / "auth_users.db"
TRAFFIC_DB_PATH: Final = Path(__file__).resolve().parent.parent / "cache" / "traffic_monitoring.db"


@dataclass(frozen=True)
class User:
    username: str
    role: str


def _password_digest(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()


def _ensure_auth_database() -> None:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUTH_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL
            )
            """
        )


def _registered_user(username: str) -> User | None:
    try:
        _ensure_auth_database()
        with sqlite3.connect(AUTH_DB_PATH) as connection:
            row = connection.execute("SELECT role FROM users WHERE username = ?", (username,)).fetchone()
        return User(username=username, role=row[0]) if row else None
    except sqlite3.Error:
        return None


def _configured_users() -> dict[str, tuple[str, str]]:
    """Read username/password hashes from env; demo values are intentionally non-production."""
    salt = os.getenv("HTI_AUTH_SALT", "local-development-salt")
    return {
        "admin_hyd": (ROLE_AUTHORITIES, os.getenv("HTI_AUTH_AUTHORITY_HASH", _password_digest("Tmc@Secure2026", salt))),
        "fleet_ops": (ROLE_FLEET, os.getenv("HTI_AUTH_FLEET_HASH", _password_digest("Fleet@Route2026", salt))),
        "research_dev": (ROLE_RESEARCH, os.getenv("HTI_AUTH_RESEARCH_HASH", _password_digest("Gwn@Predict2026", salt))),
        "security_ops": (ROLE_SECURITY, os.getenv("HTI_AUTH_SECURITY_HASH", _password_digest("Secure@Road2026", salt))),
    }


def ensure_traffic_database() -> None:
    """Create a local database for road status snapshots used by the monitoring dashboard."""
    TRAFFIC_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(TRAFFIC_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS road_status (
                segment_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                congestion_level REAL NOT NULL,
                avg_speed_kmh REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def save_road_status_snapshot(metrics: dict[tuple[str, str, str], dict[str, float]]) -> None:
    ensure_traffic_database()
    rows = []
    for (u, v, key), values in metrics.items():
        segment_id = str(values.get("segment_id", key))
        congestion = float(values.get("vc_ratio", values.get("congestion", 0.0)))
        speed = float(values.get("speed", 0.0))
        if congestion >= 0.8:
            status = "congested"
        elif congestion >= 0.45:
            status = "moderate"
        else:
            status = "free"
        rows.append((segment_id, status, congestion, speed, __import__("datetime").datetime.utcnow().isoformat()))
    if not rows:
        return
    with sqlite3.connect(TRAFFIC_DB_PATH) as connection:
        connection.executemany(
            """
            INSERT INTO road_status(segment_id, status, congestion_level, avg_speed_kmh, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(segment_id)
            DO UPDATE SET status = excluded.status, congestion_level = excluded.congestion_level, avg_speed_kmh = excluded.avg_speed_kmh, updated_at = excluded.updated_at
            """,
            rows,
        )


def _ensure_state() -> None:
    st.session_state.setdefault("hti_user", None)
    st.session_state.setdefault("hti_login_error", "")
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("user_role", None)
    st.session_state.setdefault("username", None)


def init_auth_state() -> None:
    """Initialize compatibility keys used by the multi-role login contract."""
    _ensure_state()


def authenticate(username: str, password: str) -> User | None:
    username = username.strip().lower()
    configured = _configured_users().get(username)
    if configured is not None:
        role, expected_hash = configured
        actual_hash = _password_digest(password, os.getenv("HTI_AUTH_SALT", "local-development-salt"))
        if hmac.compare_digest(actual_hash, expected_hash):
            return User(username=username, role=role)
        return None
    try:
        _ensure_auth_database()
        with sqlite3.connect(AUTH_DB_PATH) as connection:
            row = connection.execute(
                "SELECT role, password_hash, password_salt FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row and hmac.compare_digest(_password_digest(password, row[2]), row[1]):
            return User(username=username, role=row[0])
    except sqlite3.Error:
        return None
    return None


def register_user(username: str, password: str, confirmation: str, role: str) -> tuple[bool, str]:
    """Create a local account with a salted password hash."""
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,31}", username):
        return False, "Username must be 3-32 characters using lowercase letters, numbers, or underscores."
    if role not in ROLES:
        return False, "Select a valid operating role."
    if len(password) < 10 or not re.search(r"[A-Z]", password) or not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must be at least 10 characters with uppercase and special characters."
    if not hmac.compare_digest(password, confirmation):
        return False, "Passwords do not match."
    if username in _configured_users():
        return False, "That username is reserved for a configured workspace account."
    try:
        _ensure_auth_database()
        salt = secrets.token_hex(16)
        with sqlite3.connect(AUTH_DB_PATH) as connection:
            connection.execute(
                "INSERT INTO users (username, role, password_hash, password_salt) VALUES (?, ?, ?, ?)",
                (username, role, _password_digest(password, salt), salt),
            )
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username is already registered."
    except sqlite3.Error:
        return False, "Registration storage is temporarily unavailable."


def current_user() -> User | None:
    _ensure_state()
    user = st.session_state.get("hti_user")
    if user is None and st.session_state.get("authenticated"):
        role = st.session_state.get("user_role")
        username = st.session_state.get("username")
        if role and username:
            user = User(username=username, role=role)
            st.session_state["hti_user"] = user
    return user


def _set_authenticated(user: User) -> None:
    st.session_state["hti_user"] = user
    st.session_state["authenticated"] = True
    st.session_state["user_role"] = user.role
    st.session_state["username"] = user.username


def render_login_page() -> None:
    """Render the glassmorphism login gateway for the three operating roles."""
    _ensure_state()
    st.markdown(
        """
        <div class="hero-shell" style="margin-bottom:1.5rem;">
          <div>
            <div class="eyebrow">Secure gateway</div>
            <h1 class="hero-title" style="font-size:2.2rem !important;">Secure workspace sign-in</h1>
            <div class="hero-sub">Access your designated spatial analytics workspace.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.6, 1.1], gap="large")
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        login_tab, register_tab = st.tabs(["Sign in", "Register"])
        with login_tab:
            with st.form("hti-login", clear_on_submit=False):
                username = st.text_input("Username", placeholder="admin_hyd, fleet_ops, or research_dev", autocomplete="username")
                password = st.text_input("Password", type="password", placeholder="Configured workspace password", autocomplete="current-password")
                submitted = st.form_submit_button("Sign in", width="stretch")
            if submitted:
                authenticated = authenticate(username, password)
                if authenticated is None:
                    st.session_state["hti_login_error"] = "Invalid credentials."
                else:
                    _set_authenticated(authenticated)
                    st.session_state["hti_login_error"] = ""
                    st.rerun()
        with register_tab:
            st.caption("Create a local workspace account for development or pilot use.")
            with st.form("hti-register", clear_on_submit=True):
                new_username = st.text_input("New username", placeholder="your_team_name")
                new_role = st.selectbox("Workspace role", ROLES)
                new_password = st.text_input("New password", type="password")
                confirmation = st.text_input("Confirm password", type="password")
                register_submitted = st.form_submit_button("Create account", width="stretch")
            if register_submitted:
                success, message = register_user(new_username, new_password, confirmation, new_role)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        if st.session_state.get("hti_login_error"):
            st.error(st.session_state["hti_login_error"])
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass-card"><h3>Workspace directory</h3>', unsafe_allow_html=True)
        st.markdown(
            """
                        <p style="font-size:.85rem;color:#9ab1c6">Use one of these configured demo workspaces, or create a new account in the Register tab.</p>
                        <div class="route-item"><span class="route-badge">Municipal & Infrastructure Authorities</span><br><b>User:</b> <code>admin_hyd</code><br><b>Pass:</b> <code>Tmc@Secure2026</code><br><b>Audience:</b> TMC operators, planners, emergency dispatchers</div>
                        <div class="route-item"><span class="route-badge">Commercial Fleet & Transit Operations</span><br><b>User:</b> <code>fleet_ops</code><br><b>Pass:</b> <code>Fleet@Route2026</code><br><b>Audience:</b> logistics, TSRTC, ride-hailing operations</div>
                        <div class="route-item"><span class="route-badge">Transportation Researchers & Developers</span><br><b>User:</b> <code>research_dev</code><br><b>Pass:</b> <code>Gwn@Predict2026</code><br><b>Audience:</b> data scientists, ML practitioners, reviewers</div>
                        <div class="route-item"><span class="route-badge">Traffic Monitoring & Security</span><br><b>User:</b> <code>security_ops</code><br><b>Pass:</b> <code>Secure@Road2026</code><br><b>Audience:</b> police, traffic monitoring teams, emergency security desk</div>
                        <h4 style="margin-top:1.2rem;color:#ebf6ff;font-size:.9rem">Password complexity</h4>
                        <ul style="font-size:.78rem;color:#9ab1c6;padding-left:1.2rem">
                            <li>At least 10 characters</li>
                            <li>One uppercase letter</li>
                            <li>One special character</li>
                        </ul>
            <p style="font-size:.78rem;color:#9ab1c6">Production deployments should provide password hashes through HTI_AUTH_*_HASH environment variables.</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


def login_form() -> User | None:
    _ensure_state()
    user = current_user()
    if user is not None:
        return user
    render_login_page()
    return None


def logout() -> None:
    st.session_state["hti_user"] = None
    st.session_state["hti_login_error"] = ""
    st.session_state["authenticated"] = False
    st.session_state["user_role"] = None
    st.session_state["username"] = None
    st.rerun()


def has_role(user: User | None, *roles: str) -> bool:
    return user is not None and user.role in roles


def role_guard(user: User | None, *roles: str) -> bool:
    if not has_role(user, *roles):
        st.warning("This workspace is restricted to the selected operating role.")
        return False
    return True
