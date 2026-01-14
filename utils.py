import re
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = s.split()
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)
