import re

def normalize_address(address: str) -> str:
    if not isinstance(address, str):
        return ""
    addr = address.lower()
    # Remove punctuation
    addr = re.sub(r'[^\w\s]', ' ', addr)
    # Common standardizations
    addr = re.sub(r'\b(st|str)\b', 'street', addr)
    addr = re.sub(r'\brd\b', 'road', addr)
    addr = re.sub(r'\bave\b', 'avenue', addr)
    addr = re.sub(r'\bblvd\b', 'boulevard', addr)
    addr = re.sub(r'\bdr\b', 'drive', addr)
    # Remove extra spaces
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr
