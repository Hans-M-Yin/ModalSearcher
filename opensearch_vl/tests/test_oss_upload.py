"""Exercise the uploader contract without writing to object storage."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from opensearch_vl.opensearch_infer import upload


class OSSUploadTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / "test.jpg"
        self.image.write_bytes(b"image-data")
        self.env = {
            "OSS_ACCESS_KEY_ID": "test-access-id",
            "OSS_ACCESS_KEY_SECRET": "test-access-secret",
            "OSS_ENDPOINT": "oss-region.aliyuncs.com",
            "OSS_BUCKET_NAME": "example-bucket",
        }
        self.uploads = []
        self.bucket = mock.Mock()
        self.bucket.put_object.side_effect = lambda key, file, headers: self.uploads.append(
            (key, file.read(), headers)
        )
        self.auth = mock.Mock(return_value=object())
        self.bucket_factory = mock.Mock(return_value=self.bucket)
        self.sdk = SimpleNamespace(Auth=self.auth, Bucket=self.bucket_factory)

    def call_upload(self, filename="image.jpg", use_direct_url=True, extra_env=None):
        with mock.patch.dict("os.environ", {**self.env, **(extra_env or {})}, clear=True):
            with mock.patch.dict(sys.modules, {"oss2": self.sdk}):
                return upload.upload_cos(
                    str(self.image), filename, "20261026", "evaluation", "example-user", use_direct_url
                )

    def test_upload_uses_configured_endpoint_bucket_and_credentials(self):
        key, url = self.call_upload()
        self.auth.assert_called_once_with("test-access-id", "test-access-secret")
        self.bucket_factory.assert_called_once_with(
            self.auth.return_value, "https://oss-region.aliyuncs.com", "example-bucket"
        )
        self.assertEqual(key, "vision_deepresearch/20261026/evaluation/example-user/image.jpg")
        self.assertEqual(url, "https://example-bucket.oss-region.aliyuncs.com/" + key)
        self.assertEqual(self.uploads[0][:2], (key, b"image-data"))
        self.assertEqual(self.uploads[0][2]["Content-Type"], "image/jpeg")

    def test_public_domain_prefix_and_unicode_filename(self):
        key, url = self.call_upload(
            filename="folder/图像 1.jpg",
            extra_env={"OSS_PUBLIC_BASE_URL": "https://images.example.com/media/"},
        )
        self.assertTrue(key.endswith("/图像 1.jpg"))
        self.assertEqual(
            url,
            "https://images.example.com/media/vision_deepresearch/20261026/"
            "evaluation/example-user/%E5%9B%BE%E5%83%8F%201.jpg",
        )

    def test_configured_endpoint_scheme_is_preserved(self):
        _, url = self.call_upload(extra_env={"OSS_ENDPOINT": "http://storage.example.com/"})
        self.assertTrue(url.startswith("http://example-bucket.storage.example.com/"))
        self.assertEqual(self.bucket_factory.call_args.args[1], "http://storage.example.com")

    def test_upload_can_return_only_the_object_key(self):
        key, url = self.call_upload(use_direct_url=False)
        self.assertTrue(key.endswith("/image.jpg"))
        self.assertIsNone(url)
        self.assertEqual(len(self.uploads), 1)

    def test_missing_endpoint_fails_before_upload(self):
        with self.assertRaisesRegex(RuntimeError, "OSS_ENDPOINT"):
            self.call_upload(extra_env={"OSS_ENDPOINT": ""})
        self.bucket_factory.assert_not_called()
        self.assertFalse(self.uploads)


if __name__ == "__main__":
    unittest.main()
