import base64
from pathlib import Path
from typing import List, Union

SUPPORTED_LANGUAGES = ['hindi', 'tamil', 'indonesian', 'japanese', 'vietnamese']


def validate_languages(languages: Union[str, List[str]]) -> List[str]:
    """Validate and return list of languages to evaluate."""
    if len(languages) == 1 and languages[0].lower() == 'all':
        return SUPPORTED_LANGUAGES
    input_languages = [lang.lower() for lang in languages]
    invalid_languages = [lang for lang in input_languages if lang not in SUPPORTED_LANGUAGES]
    if invalid_languages:
        raise ValueError(f"Unsupported languages: {', '.join(invalid_languages)}\n"
                         f"Supported languages are: {', '.join(SUPPORTED_LANGUAGES)}")
    return input_languages


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string for API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from image file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')
