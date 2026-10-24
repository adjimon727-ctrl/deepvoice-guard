import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.requests import Request

import api


class ScanAccessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.original_database_path = api.accounts.DATABASE_PATH
        self.original_image_classifier = api.image_classifier
        api.accounts.DATABASE_PATH = Path(self.temp_directory.name) / "test.db"
        api.accounts.initialize_database()
        account = api.accounts.create_account("scan@example.com", "StrongPassword123!")
        self.account = account
        self.token = api.accounts.create_session(account["id"])

    async def asyncTearDown(self):
        api.accounts.DATABASE_PATH = self.original_database_path
        api.image_classifier = self.original_image_classifier
        self.temp_directory.cleanup()

    def request(self, token=None):
        headers = []
        if token:
            headers.append((b"cookie", f"deepvoice_session={token}".encode()))
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/scan",
                "headers": headers,
                "query_string": b"",
                "scheme": "http",
                "server": ("localhost", 8000),
                "client": ("127.0.0.1", 12345),
            }
        )

    async def test_scan_requires_login(self):
        upload = UploadFile(filename="test.jpg", file=io.BytesIO(b"image"))

        with self.assertRaises(HTTPException) as raised:
            await api.scan_media(self.request(), upload, "image")

        self.assertEqual(raised.exception.status_code, 401)

    async def test_free_plan_cannot_scan_audio_via_api(self):
        upload = UploadFile(filename="test.wav", file=io.BytesIO(b"audio"))

        with self.assertRaises(HTTPException) as raised:
            await api.scan_media(self.request(self.token), upload, "audio")

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(api.accounts.get_account_usage(self.account["id"])["usage"]["scans"], 0)

    async def test_free_plan_can_analyze_image_and_spend_quota(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (4, 4), color="red").save(image_buffer, format="PNG")
        upload = UploadFile(filename="test.png", file=io.BytesIO(image_buffer.getvalue()))
        api.image_classifier = lambda image: [
            {"label": "Deepfake", "score": 0.8},
            {"label": "Real one", "score": 0.2},
        ]

        result = await api.scan_media(self.request(self.token), upload, "image")

        self.assertEqual(result["analyses"], {"fake": 80.0, "real": 20.0})
        self.assertEqual(result["verdict_principal"], "fake")
        self.assertEqual(api.accounts.get_account_usage(self.account["id"])["usage"]["scans"], 1)

    async def test_verified_payment_activates_subscription(self):
        api.accounts.create_pending_payment(
            "verified-reference", self.account["id"], "pro"
        )
        with patch.object(
            api.billing,
            "verify_transaction",
            return_value={
                "status": "success",
                "reference": "verified-reference",
                "amount": 5_000,
                "currency": "XOF",
            },
        ):
            await api._confirm_payment("verified-reference")

        usage = api.accounts.get_account_usage(self.account["id"])
        self.assertEqual(usage["account"]["plan"], "pro")

    async def test_payment_with_wrong_amount_does_not_activate_subscription(self):
        api.accounts.create_pending_payment(
            "wrong-amount-reference", self.account["id"], "pro"
        )
        with patch.object(
            api.billing,
            "verify_transaction",
            return_value={
                "status": "success",
                "reference": "wrong-amount-reference",
                "amount": 1,
                "currency": "XOF",
            },
        ):
            with self.assertRaises(HTTPException) as raised:
                await api._confirm_payment("wrong-amount-reference")

        self.assertEqual(raised.exception.status_code, 400)
        usage = api.accounts.get_account_usage(self.account["id"])
        self.assertEqual(usage["account"]["plan"], "free")

    async def test_payment_callback_uses_configured_root_url(self):
        with patch.dict(os.environ, {"PUBLIC_BASE_URL": "https://deepvoice.example"}):
            self.assertEqual(
                api._public_base_url(),
                "https://deepvoice.example",
            )
        with patch.dict(
            os.environ, {"PUBLIC_BASE_URL": "https://deepvoice.example/attacker"}
        ):
            with self.assertRaises(HTTPException) as raised:
                api._public_base_url()
        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
