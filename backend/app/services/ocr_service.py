import logging
import io
from PIL import Image
from backend.app.core.config import settings

logger = logging.getLogger("phishing_platform")

# Optional import of pytesseract to avoid crashes if it's missing entirely
try:
    import pytesseract
    HAS_PYTESSERACT = True
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
except ImportError:
    HAS_PYTESSERACT = False

class OCRService:
    @staticmethod
    def extract_text_from_bytes(image_bytes: bytes) -> str:
        """Extracts text from image bytes using Tesseract OCR."""
        if not HAS_PYTESSERACT:
            logger.warning("pytesseract library is not installed.")
            return "[OCR Error: pytesseract library is not installed on the backend]"

        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Perform OCR
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            error_str = str(e)
            logger.error(f"OCR failed: {error_str}")
            if "tesseract is not installed" in error_str.lower() or "no such file or directory" in error_str.lower():
                return ("[OCR Warning: Tesseract OCR engine is not installed or configured on the host system. "
                        "Please install tesseract-ocr binaries and set its path in the environment.]")
            return f"[OCR Error: {error_str}]"
