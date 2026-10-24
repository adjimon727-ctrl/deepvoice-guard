import hashlib
import hmac
import calendar
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path


DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(Path(__file__).resolve().parent / "deepvoice_guard.db"))
)
SESSION_COOKIE_NAME = "deepvoice_session"
SESSION_DURATION_SECONDS = 60 * 60 * 24 * 14
PASSWORD_ITERATIONS = 600_000
EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,63}$")
PLANS = {
    "free": {
        "name": "Gratuit",
        "price_xof": 0,
        "monthly_scans": 10,
        "modalities": ["image"],
    },
    "pro": {
        "name": "Pro",
        "price_xof": 5_000,
        "monthly_scans": 300,
        "modalities": ["image", "audio", "video"],
    },
    "business": {
        "name": "Business",
        "price_xof": 25_000,
        "monthly_scans": 2_000,
        "modalities": ["image", "audio", "video"],
    },
}


class PlanRestrictionError(Exception):
    pass


class QuotaExceededError(Exception):
    pass


@contextmanager
def _connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    with _connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT 'free'
                    CHECK (plan IN ('free', 'pro', 'business')),
                plan_expires_at INTEGER,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sessions_account_id_idx ON sessions(account_id);
            CREATE TABLE IF NOT EXISTS monthly_usage (
                account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                month TEXT NOT NULL,
                scans INTEGER NOT NULL DEFAULT 0 CHECK (scans >= 0),
                PRIMARY KEY (account_id, month)
            );
            CREATE TABLE IF NOT EXISTS payments (
                reference TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                plan TEXT NOT NULL CHECK (plan IN ('pro', 'business')),
                amount_xof INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'success'))
            );
            """
        )


def validate_credentials(email, password):
    email = email.strip().lower()
    if not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Adresse e-mail invalide.")
    if len(password) < 12 or len(password) > 128:
        raise ValueError("Le mot de passe doit contenir entre 12 et 128 caractères.")
    return email


def _hash_password(password, salt):
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
        dklen=32,
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${derived_key.hex()}"


def _verify_password(password, encoded_hash):
    try:
        algorithm, iterations, salt_hex, expected_hash = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256" or int(iterations) != PASSWORD_ITERATIONS:
            return False
        actual_hash = _hash_password(password, bytes.fromhex(salt_hex)).rsplit("$", 1)[1]
        return hmac.compare_digest(actual_hash, expected_hash)
    except (ValueError, TypeError):
        return False


DUMMY_PASSWORD_HASH = _hash_password("dummy-password-never-used", b"\x00" * 16)


def _public_account(row):
    return {
        "id": row["id"],
        "email": row["email"],
        "plan": row["plan"],
        "plan_expires_at": row["plan_expires_at"],
    }


def create_account(email, password):
    email = validate_credentials(email, password)
    salt = secrets.token_bytes(16)
    account_id = secrets.token_urlsafe(18)
    now = int(time.time())
    with _connect() as connection:
        connection.execute(
            "INSERT INTO accounts (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (account_id, email, _hash_password(password, salt), now),
        )
        row = connection.execute(
            "SELECT id, email, plan, plan_expires_at FROM accounts WHERE id = ?",
            (account_id,),
        ).fetchone()
    return _public_account(row)


def create_session(account_id):
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at = int(time.time()) + SESSION_DURATION_SECONDS
    with _connect() as connection:
        connection.execute(
            "INSERT INTO sessions (token_hash, account_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, account_id, expires_at),
        )
    return token


def authenticate(email, password):
    normalized_email = email.strip().lower()
    if len(normalized_email) > 320 or len(password) > 128:
        return None
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, email, password_hash, plan, plan_expires_at
            FROM accounts WHERE email = ?
            """,
            (normalized_email,),
        ).fetchone()

    if row is None:
        _verify_password(password, DUMMY_PASSWORD_HASH)
        return None
    if not _verify_password(password, row["password_hash"]):
        return None
    return _public_account(row), create_session(row["id"])


def get_account_for_session(token):
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = int(time.time())
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT a.id, a.email, a.plan, a.plan_expires_at
            FROM sessions s JOIN accounts a ON a.id = s.account_id
            WHERE s.token_hash = ? AND s.expires_at > ?
            """,
            (token_hash, now),
        ).fetchone()
        if row is None:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            return None
    return _public_account(row)


def revoke_session(token):
    if not token:
        return
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _connect() as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def _current_month():
    return time.strftime("%Y-%m", time.gmtime())


def _active_plan(row, now):
    if row["plan"] != "free" and (
        row["plan_expires_at"] is None or row["plan_expires_at"] <= now
    ):
        return "free"
    return row["plan"]


def reserve_scan(account_id, mode):
    month = _current_month()
    now = int(time.time())
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        account = connection.execute(
            "SELECT plan, plan_expires_at FROM accounts WHERE id = ?",
            (account_id,),
        ).fetchone()
        if account is None:
            raise PlanRestrictionError("Compte introuvable.")

        plan = _active_plan(account, now)
        if plan == "free" and account["plan"] != "free":
            connection.execute(
                "UPDATE accounts SET plan = 'free', plan_expires_at = NULL WHERE id = ?",
                (account_id,),
            )
        plan_details = PLANS[plan]
        if mode not in plan_details["modalities"]:
            raise PlanRestrictionError(
                "Cette analyse est réservée aux offres Pro et Business."
            )

        usage = connection.execute(
            "SELECT scans FROM monthly_usage WHERE account_id = ? AND month = ?",
            (account_id, month),
        ).fetchone()
        current_scans = usage["scans"] if usage else 0
        if current_scans >= plan_details["monthly_scans"]:
            raise QuotaExceededError(
                "Le quota mensuel de votre formule est atteint."
            )

        connection.execute(
            """
            INSERT INTO monthly_usage (account_id, month, scans) VALUES (?, ?, 1)
            ON CONFLICT(account_id, month) DO UPDATE SET scans = scans + 1
            """,
            (account_id, month),
        )


def refund_scan(account_id):
    with _connect() as connection:
        connection.execute(
            """
            UPDATE monthly_usage SET scans = MAX(scans - 1, 0)
            WHERE account_id = ? AND month = ?
            """,
            (account_id, _current_month()),
        )


def get_account_usage(account_id):
    month = _current_month()
    now = int(time.time())
    with _connect() as connection:
        account = connection.execute(
            "SELECT id, email, plan, plan_expires_at FROM accounts WHERE id = ?",
            (account_id,),
        ).fetchone()
        if account is None:
            return None
        plan = _active_plan(account, now)
        if plan == "free" and account["plan"] != "free":
            connection.execute(
                "UPDATE accounts SET plan = 'free', plan_expires_at = NULL WHERE id = ?",
                (account_id,),
            )
        usage = connection.execute(
            "SELECT scans FROM monthly_usage WHERE account_id = ? AND month = ?",
            (account_id, month),
        ).fetchone()
    details = PLANS[plan]
    return {
        "account": {
            "id": account["id"],
            "email": account["email"],
            "plan": plan,
            "plan_expires_at": account["plan_expires_at"] if plan != "free" else None,
        },
        "usage": {
            "month": month,
            "scans": usage["scans"] if usage else 0,
            "monthly_limit": details["monthly_scans"],
            "remaining": max(
                0,
                details["monthly_scans"] - (usage["scans"] if usage else 0),
            ),
        },
    }


def create_pending_payment(reference, account_id, plan):
    if plan not in ("pro", "business"):
        raise ValueError("Offre payante inconnue.")
    amount_xof = PLANS[plan]["price_xof"]
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO payments (reference, account_id, plan, amount_xof)
            VALUES (?, ?, ?, ?)
            """,
            (reference, account_id, plan, amount_xof),
        )
    return amount_xof


def get_payment(reference):
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT reference, account_id, plan, amount_xof, status
            FROM payments WHERE reference = ?
            """,
            (reference,),
        ).fetchone()
    return dict(row) if row else None


def activate_payment(reference):
    now = int(time.time())
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        payment = connection.execute(
            """
            SELECT account_id, plan, status
            FROM payments WHERE reference = ?
            """,
            (reference,),
        ).fetchone()
        if payment is None:
            return False
        if payment["status"] == "success":
            return True

        account = connection.execute(
            "SELECT plan, plan_expires_at FROM accounts WHERE id = ?",
            (payment["account_id"],),
        ).fetchone()
        if account is None:
            return False

        current_expiry = account["plan_expires_at"] or 0
        start = max(now, current_expiry) if account["plan"] != "free" else now
        start_date = datetime.fromtimestamp(start, timezone.utc)
        month = start_date.month % 12 + 1
        year = start_date.year + (start_date.month // 12)
        day = min(start_date.day, calendar.monthrange(year, month)[1])
        expires_at = int(
            start_date.replace(year=year, month=month, day=day).timestamp()
        )
        connection.execute(
            """
            UPDATE accounts SET plan = ?, plan_expires_at = ? WHERE id = ?
            """,
            (payment["plan"], expires_at, payment["account_id"]),
        )
        connection.execute(
            "UPDATE payments SET status = 'success' WHERE reference = ?",
            (reference,),
        )
    return True
