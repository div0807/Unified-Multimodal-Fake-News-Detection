from transformers import BertTokenizer

_tokenizer = None


def _get_tokenizer() -> BertTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return _tokenizer


def process_text(text: str, max_length: int = 128) -> dict:
    """
    Tokenize a string for BERT.

    Returns:
        dict with 'input_ids' and 'attention_mask', both shape (1, max_length).
    """
    tokenizer = _get_tokenizer()
    return tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )