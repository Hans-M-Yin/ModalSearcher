"""Regression tests for the Serper client and explicit API credentials."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from unittest import mock

from synthesis.search_client import (
    SerperSearchClient,
    _is_serper_auth_failure,
    _is_serper_credit_exhausted,
)


class _Response:
    def __init__(self, payload: bytes, status_code: int = 200) -> None:
        self._payload = payload
        self._status_code = status_code

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return self._payload

    def getcode(self) -> int:
        return self._status_code


class _FakeOpener:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def open(self, request, *, timeout: float):
        del timeout
        key = request.get_header("X-api-key")
        self.keys.append(key)
        if key == "bad-key":
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"message":"Not enough credits","statusCode":400}'),
            )
        return _Response(b'{"images":[]}')


class SerperClientTests(unittest.TestCase):

    def test_default_official_url_uses_the_configured_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            "os.environ",
            {
                "SERPER_API_KEY": "configured-key",
            },
            clear=False,
        ):
            state_path = Path(tmpdir) / "serper_pool_state.json"
            client = SerperSearchClient(
                api_keys=["local-real-key"],
                pool_state_path=state_path,
            )
            opener = _FakeOpener()
            client._url_opener = opener

            client.search_text("client test", limit=1)

            self.assertEqual(client.search_url, "https://google.serper.dev/search")

            self.assertIsNone(client.key_pool)
            self.assertEqual(opener.keys, ["configured-key"])
            self.assertFalse(state_path.exists())

    def test_not_enough_credits_is_credit_exhaustion_but_rate_limit_is_not(self) -> None:
        self.assertTrue(
            _is_serper_credit_exhausted(
                status_code=400,
                response_body='{"message":"Not enough credits","statusCode":400}',
            )
        )
        self.assertFalse(
            _is_serper_credit_exhausted(
                status_code=429,
                response_body='{"message":"Not enough credits"}',
            )
        )

    def test_unauthorized_is_an_auth_failure_but_generic_403_is_not(self) -> None:
        self.assertTrue(
            _is_serper_auth_failure(
                status_code=403,
                response_body='{"message":"Unauthorized.","statusCode":403}',
            )
        )
        self.assertTrue(
            _is_serper_auth_failure(
                status_code=401,
                response_body='{"message":"Invalid API key"}',
            )
        )
        self.assertFalse(
            _is_serper_auth_failure(
                status_code=403,
                response_body='{"message":"Forbidden by policy"}',
            )
        )

    def test_worker_uses_one_fixed_key_without_local_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            "os.environ", {"SERPER_API_KEY": "fixed-key"}, clear=False
        ):
            state_path = Path(tmpdir) / "serper_pool_state.json"
            client = SerperSearchClient(
                api_keys=["bad-key", "good-key"],
                images_url="https://google.serper.dev/images",
                pool_state_path=state_path,
                pool_default_credits=100,
                pool_min_remaining=0,
            )
            opener = _FakeOpener()
            client._url_opener = opener

            client.search_image("Indian motorcycle first prototype", limit=10, hl="en")
            client.search_image("Indian motorcycle first prototype", limit=10, hl="en")

            self.assertEqual(opener.keys, ["fixed-key", "fixed-key"])
            self.assertFalse(state_path.exists())


if __name__ == "__main__":
    unittest.main()
