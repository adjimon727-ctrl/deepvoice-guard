import hashlib
import hmac
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PAYSTACK_API_URL = "https://api.paystack.co"


class PaystackError(Exception):
    pass


def _secret_key():
    secret = os.getenv("PAYSTACK_SECRET_KEY")
    if not secret:
        raise PaystackError("Les paiements ne sont pas encore configurés.")
    return secret


def _request(path, payload=None):
    secret = _secret_key()
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{PAYSTACK_API_URL}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PaystackError("Impossible de communiquer avec Paystack.") from exc
    if not result.get("status"):
        raise PaystackError("Paystack a refusé la demande de paiement.")
    return result.get("data", {})


def initialize_transaction(email, reference, amount_xof, callback_url):
    return _request(
        "/transaction/initialize",
        {
            "email": email,
            "amount": amount_xof,
            "currency": "XOF",
            "reference": reference,
            "callback_url": callback_url,
        },
    )


def verify_transaction(reference):
    return _request(f"/transaction/verify/{quote(reference, safe='')}")


def verify_webhook_signature(payload, signature):
    if not signature:
        return False
    expected = hmac.new(
        _secret_key().encode("utf-8"), payload, hashlib.sha512
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
