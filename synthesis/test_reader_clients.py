"""Reader behavior remains available without the private graph builder."""

import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from synthesis.wiki_text_builder import EnhancedReaderClient, RawMarkdownReaderClient


def response_with(payload: bytes) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = payload
    return response


class ReaderClientTests(unittest.TestCase):
    def test_enhanced_reader_preserves_document_fields_and_normalizes_url(self):
        payload = {"data": {"title": "Example", "content": "Evidence", "raw_markdown": "# Example"}}
        with patch(
            "synthesis.wiki_text_builder.urlopen",
            return_value=response_with(json.dumps(payload).encode()),
        ) as request:
            document = EnhancedReaderClient(base_url="http://reader.example/").read("example.com/page")

        self.assertEqual(document.url, "https://example.com/page")
        self.assertEqual(document.title, "Example")
        self.assertEqual(document.content, "Evidence")
        self.assertEqual(document.raw_markdown, "# Example")
        self.assertEqual(document.raw, payload)
        self.assertEqual(request.call_args.args[0].full_url, "http://reader.example/https://example.com/page")

    def test_enhanced_reader_retries_502_once(self):
        payload = b'{"data": {"content": "recovered"}}'
        error = HTTPError("http://reader.example", 502, "temporary", {}, None)
        with (
            patch("synthesis.wiki_text_builder.urlopen", side_effect=[error, response_with(payload)]) as request,
            patch("synthesis.wiki_text_builder.time.sleep") as sleep,
            patch("synthesis.wiki_text_builder._reader_retry_debug"),
        ):
            document = EnhancedReaderClient().read("https://example.com")

        self.assertEqual(document.content, "recovered")
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(15)

    def test_enhanced_reader_respects_disabled_retry(self):
        error = HTTPError("http://reader.example", 502, "temporary", {}, None)
        with (
            patch("synthesis.wiki_text_builder.urlopen", side_effect=error) as request,
            patch("synthesis.wiki_text_builder.time.sleep") as sleep,
        ):
            with self.assertRaises(HTTPError):
                EnhancedReaderClient(retry_502=False).read("https://example.com")

        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_raw_markdown_reader_preserves_title_extraction(self):
        for url, title in [
            ("https://en.wikipedia.org/wiki/A_B%26C", "A B&C"),
            ("https://example.com/page", None),
        ]:
            with self.subTest(url=url), patch(
                "synthesis.wiki_text_builder.urlopen", return_value=response_with(b"# Evidence")
            ):
                document = RawMarkdownReaderClient().read(url)

            self.assertEqual(document.url, url)
            self.assertEqual(document.title, title)
            self.assertEqual(document.content, "# Evidence")
            self.assertEqual(document.raw_markdown, "# Evidence")
            self.assertEqual(document.raw, {"reader": "raw_markdown_reader"})


if __name__ == "__main__":
    unittest.main()
