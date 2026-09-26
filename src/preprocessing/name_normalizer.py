import re

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    # Remove punctuation and special characters
    name = re.sub(r'[^\w\s]', ' ', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name
