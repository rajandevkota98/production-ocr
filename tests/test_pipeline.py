import unittest
from gateway.sources.loader import detect_and_validate_format, encode_payload
from gateway.services.exceptions import (
    InvalidPayloadException,
    TaskNotFoundException,
    RedisConnectionException,
)

class TestOcrPipeline(unittest.TestCase):
    def test_detect_format_pdf(self):
        pdf_bytes = b"%PDF-1.4 sample content"
        self.assertEqual(detect_and_validate_format(pdf_bytes), "pdf")

    def test_detect_format_png(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"extra data"
        self.assertEqual(detect_and_validate_format(png_bytes), "png")

    def test_detect_format_jpeg(self):
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"extra data"
        self.assertEqual(detect_and_validate_format(jpeg_bytes), "jpg")

    def test_detect_format_webp(self):
        webp_bytes = b"RIFF0000WEBP" + b"extra data"
        self.assertEqual(detect_and_validate_format(webp_bytes), "webp")

    def test_detect_format_invalid(self):
        with self.assertRaises(InvalidPayloadException):
            detect_and_validate_format(b"NOT A VALID FILE")

    def test_encode_payload(self):
        sample = b"hello world"
        encoded = encode_payload(sample)
        self.assertIsInstance(encoded, str)
        self.assertTrue(len(encoded) > 0)

    def test_custom_exceptions(self):
        exc = TaskNotFoundException("12345")
        self.assertEqual(exc.status_code, 404)
        self.assertIn("12345", exc.message)

if __name__ == "__main__":
    unittest.main()
