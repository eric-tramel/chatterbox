from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1



def drop_invalid_tokens(x):
    """Drop SoS and EoS for a tensor with shape (T,) or (B, T)."""

    def _trim(sequence):
        if SOS in sequence:
            # Ensure s is a Python int or 0-d tensor item
            s = (sequence == SOS).nonzero(as_tuple=True)[0]
            if s.numel() > 1:
                s = s[0]  # Take first occurrence
            s = s.item() + 1
        else:
            s = 0

        if EOS in sequence:
            # Ensure e is a Python int
            e = (sequence == EOS).nonzero(as_tuple=True)[0]
            if e.numel() > 1:
                e = e[0]  # Take first occurrence
            e = e.item()
        else:
            e = None
        return sequence[s:e]

    if x.dim() == 1:
        return _trim(x)

    assert x.dim() == 2, "drop_invalid_tokens expects a 1D or 2D tensor"
    trimmed = [_trim(row) for row in x]
    if not trimmed:
        return x.new_empty(0)

    max_len = max(seq.size(0) for seq in trimmed)
    padded = x.new_full((len(trimmed), max_len), EOS)
    for idx, seq in enumerate(trimmed):
        padded[idx, : seq.size(0)] = seq
    return padded
