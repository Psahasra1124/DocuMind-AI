import os
import unittest
from unittest import mock
from uuid import uuid4

import jwt
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_document_rag",
)
os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-only-jwt-secret-key-32-bytes-long",
)
os.environ.setdefault("JWT_ALGORITHM", "HS256")

from app.api.dependencies import auth as auth_module
from app.core.config import Settings


class AuthConfigRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_current_user_uses_uppercase_jwt_settings(self):
        user_id = uuid4()
        secret = "test-only-jwt-secret-key-32-bytes-long"
        token = jwt.encode(
            {"sub": str(user_id)},
            secret,
            algorithm="HS256",
        )

        auth_module.get_user_by_id = mock.AsyncMock(
            return_value=mock.Mock(id=user_id, is_active=True)
        )

        user = await auth_module.get_current_user(
            credentials=HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=token,
            ),
            db=mock.AsyncMock(),
        )

        self.assertEqual(user.id, user_id)
        self.assertTrue(user.is_active)


class SettingsSecurityTests(unittest.TestCase):
    DATABASE_URL = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/"
        "ai_document_rag"
    )
    VALID_SECRET = "test-only-jwt-secret-key-32-bytes-long"

    def test_valid_sufficiently_long_jwt_secret_is_accepted(self):
        settings = Settings(
            DATABASE_URL=self.DATABASE_URL,
            JWT_SECRET_KEY=self.VALID_SECRET,
        )

        self.assertEqual(settings.JWT_SECRET_KEY, self.VALID_SECRET)

    def test_short_jwt_secret_is_rejected(self):
        with self.assertRaises(ValidationError) as context:
            Settings(
                DATABASE_URL=self.DATABASE_URL,
                JWT_SECRET_KEY="short-secret",
            )

        self.assertIn("at least 32 characters", str(context.exception))

    def test_missing_jwt_secret_key_is_rejected(self):
        with mock.patch.dict(
            os.environ,
            {
                "DATABASE_URL": self.DATABASE_URL,
            },
            clear=True,
        ):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_missing_database_url_is_rejected(self):
        with mock.patch.dict(
            os.environ,
            {
                "JWT_SECRET_KEY": self.VALID_SECRET,
            },
            clear=True,
        ):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None)

    def test_render_postgres_url_is_normalized_for_asyncpg(self):
        settings = Settings(
            DATABASE_URL=(
                "postgres://db-user:db-password@ep-example.neon.tech/app"
                "?sslmode=require"
            ),
            JWT_SECRET_KEY=self.VALID_SECRET,
        )

        self.assertEqual(
            settings.async_database_url,
            "postgresql+asyncpg://db-user:db-password@ep-example.neon.tech/app"
            "?ssl=require",
        )

    def test_sync_migration_url_uses_psycopg2(self):
        settings = Settings(
            DATABASE_URL="postgresql://db-user:db-password@db.example/app",
            SYNC_DATABASE_URL=(
                "postgresql://migration-user:migration-password@db.example/app"
                "?sslmode=require"
            ),
            JWT_SECRET_KEY=self.VALID_SECRET,
        )

        self.assertEqual(
            settings.alembic_database_url,
            "postgresql+psycopg2://migration-user:migration-password"
            "@db.example/app?sslmode=require",
        )

    def test_async_database_url_is_converted_for_sync_migrations(self):
        settings = Settings(
            DATABASE_URL=(
                "postgresql+asyncpg://db-user:db-password@localhost/app"
            ),
            JWT_SECRET_KEY=self.VALID_SECRET,
        )

        self.assertEqual(
            settings.alembic_database_url,
            "postgresql+psycopg2://db-user:db-password@localhost/app",
        )

    def test_cloudflare_provider_requires_backend_credentials(self):
        with self.assertRaises(ValidationError) as context:
            Settings(
                DATABASE_URL=self.DATABASE_URL,
                JWT_SECRET_KEY=self.VALID_SECRET,
                LLM_PROVIDER="cloudflare",
            )

        self.assertIn("CLOUDFLARE_ACCOUNT_ID", str(context.exception))

    def test_cloudflare_provider_accepts_configured_credentials(self):
        settings = Settings(
            DATABASE_URL=self.DATABASE_URL,
            JWT_SECRET_KEY=self.VALID_SECRET,
            LLM_PROVIDER="CLOUDFLARE",
            CLOUDFLARE_ACCOUNT_ID="test-account-id",
            CLOUDFLARE_API_TOKEN="test-api-token",
        )

        self.assertEqual(settings.LLM_PROVIDER, "cloudflare")
        self.assertEqual(
            settings.CLOUDFLARE_MODEL,
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        )


if __name__ == "__main__":
    unittest.main()
