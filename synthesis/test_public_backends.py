"""Check public service selection without external requests or provider SDKs."""
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from synthesis.firecrawl_client import (
    FirecrawlBrowserSessionManager,
    FirecrawlClient,
    _FixedFirecrawlKeyProvider,
)
from synthesis.sft import tools


class PublicBackendTests(unittest.TestCase):
    def test_scrape_constructs_the_official_sdk_with_configured_key(self):
        sdk = mock.Mock()
        sdk.scrape.return_value = {
            "success": True,
            "data": {"markdown": "public page", "metadata": {"statusCode": 200}},
        }
        factory = mock.Mock(return_value=sdk)
        with mock.patch.dict(sys.modules, {"firecrawl": SimpleNamespace(Firecrawl=factory)}):
            result = FirecrawlClient(api_key="configured-key").scrape(
                "https://example.com/page", only_main_content=True, max_age=1200, formats=["markdown"]
            )
        factory.assert_called_once_with(api_key="configured-key", api_url="https://api.firecrawl.dev")
        sdk.scrape.assert_called_once_with(
            "https://example.com/page", only_main_content=True, max_age=1200, formats=["markdown"]
        )
        self.assertEqual(result["data"]["markdown"], "public page")

    def test_browser_constructs_the_official_sdk_with_configured_timeout(self):
        provider = _FixedFirecrawlKeyProvider(explicit_key="configured-key")
        factory = mock.Mock()
        with TemporaryDirectory() as directory:
            manager = FirecrawlBrowserSessionManager(
                key_pool=provider, state_path=Path(directory) / "sessions.json", api_timeout_s=180
            )
            with mock.patch.dict(sys.modules, {"firecrawl.v2": SimpleNamespace(FirecrawlClient=factory)}):
                app = manager._app(provider.key_id_value)
        factory.assert_called_once_with(
            api_key="configured-key", api_url="https://api.firecrawl.dev", timeout=180
        )
        self.assertIs(app, factory.return_value)

    def test_inference_provider_defaults_use_public_endpoints(self):
        path = Path(__file__).resolve().parents[1] / "opensearch_vl/opensearch_infer/config.py"
        with mock.patch.dict("os.environ", {}, clear=True):
            config = runpy.run_path(str(path))
        self.assertEqual(config["SERPER_SEARCH_URL"], "https://google.serper.dev/search")
        self.assertEqual(config["SERPER_IMAGES_URL"], "https://google.serper.dev/images")
        self.assertEqual(config["SERPER_LENS_URL"], "https://google.serper.dev/lens")
        self.assertEqual(config["JINA_READER_URL"], "https://r.jina.ai/")
        self.assertEqual(config["ENHANCED_READER_URL"], "https://r.jina.ai/")

    def test_shared_tools_default_to_the_public_reader_branch(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(tools, "_read_via_jina", return_value="public reader content") as read:
                with mock.patch.object(tools, "EnhancedReaderClient") as enhanced:
                    document = tools._read_document("https://example.com/page")
        read.assert_called_once_with("https://example.com/page")
        enhanced.assert_not_called()
        self.assertEqual(document["content"], "public reader content")
        self.assertEqual(document["raw_markdown"], "public reader content")


if __name__ == "__main__":
    unittest.main()
