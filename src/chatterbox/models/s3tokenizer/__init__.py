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
            s = (sequence == SOS).nonzero(as_tuple=True)[0].squeeze(0) + 1
        else:
            s = 0

        if EOS in sequence:
            e = (sequence == EOS).nonzero(as_tuple=True)[0].squeeze(0)
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
