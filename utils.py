import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9]+", " ", text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", " ", text)
    text = re.sub(r"[^a-zA-z.!?']", " ", text)
    text = re.sub(r" +", " ", text)
    return text
