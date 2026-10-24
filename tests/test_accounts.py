import tempfile
import unittest
from pathlib import Path

import accounts


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.original_database_path = accounts.DATABASE_PATH
        accounts.DATABASE_PATH = Path(self.temp_directory.name) / "test.db"
        accounts.initialize_database()

    def tearDown(self):
        accounts.DATABASE_PATH = self.original_database_path
        self.temp_directory.cleanup()

    def test_register_login_and_revoke_session(self):
        account = accounts.create_account("Test@Example.com", "StrongPassword123!")
        authenticated = accounts.authenticate("test@example.com", "StrongPassword123!")

        self.assertIsNotNone(authenticated)
        logged_in_account, token = authenticated
        self.assertEqual(account["id"], logged_in_account["id"])
        self.assertEqual(account, accounts.get_account_for_session(token))

        accounts.revoke_session(token)
        self.assertIsNone(accounts.get_account_for_session(token))

    def test_invalid_credentials_are_rejected(self):
        with self.assertRaises(ValueError):
            accounts.create_account("not-an-email", "StrongPassword123!")
        with self.assertRaises(ValueError):
            accounts.create_account("test@example.com", "short")

        accounts.create_account("test@example.com", "StrongPassword123!")
        self.assertIsNone(accounts.authenticate("test@example.com", "WrongPassword123!"))

    def test_free_plan_enforces_modality_and_monthly_quota(self):
        account = accounts.create_account("test@example.com", "StrongPassword123!")

        with self.assertRaises(accounts.PlanRestrictionError):
            accounts.reserve_scan(account["id"], "audio")

        for _ in range(10):
            accounts.reserve_scan(account["id"], "image")

        with self.assertRaises(accounts.QuotaExceededError):
            accounts.reserve_scan(account["id"], "image")

    def test_successful_payment_activates_plan_once(self):
        account = accounts.create_account("test@example.com", "StrongPassword123!")
        amount = accounts.create_pending_payment("test-reference", account["id"], "pro")

        self.assertEqual(amount, 5_000)
        self.assertTrue(accounts.activate_payment("test-reference"))
        self.assertTrue(accounts.activate_payment("test-reference"))

        usage = accounts.get_account_usage(account["id"])
        self.assertEqual(usage["account"]["plan"], "pro")
        self.assertGreater(usage["account"]["plan_expires_at"], 0)

    def test_unknown_paid_plan_cannot_create_payment(self):
        account = accounts.create_account("test@example.com", "StrongPassword123!")
        with self.assertRaises(ValueError):
            accounts.create_pending_payment("invalid-reference", account["id"], "free")


if __name__ == "__main__":
    unittest.main()
