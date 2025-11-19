import copy

import torch
import types

from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3 import llama_configs
from chatterbox.mtl_tts import (
    ChatterboxMultilingualTTS,
    Conditionals as MTLConditionals,
    _SAMPLES_PER_MEL_FRAME as MTL_SAMPLES_PER_MEL_FRAME,
    _SAMPLES_PER_SPEECH_TOKEN as MTL_SAMPLES_PER_SPEECH_TOKEN,
)


TINY_LLAMA_CONFIG = dict(
    vocab_size=256,
    max_position_embeddings=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    attn_implementation="eager",
    head_dim=16,
    tie_word_embeddings=False,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    rope_theta=10000.0,
    torch_dtype="float32",
    use_cache=True,
)


def _ensure_tiny_config():
    if "TinyTest" not in llama_configs.LLAMA_CONFIGS:
        llama_configs.LLAMA_CONFIGS["TinyTest"] = TINY_LLAMA_CONFIG


def _make_tiny_hp():
    _ensure_tiny_config()
    hp = T3Config.multilingual()
    hp.llama_config_name = "TinyTest"
    hp.max_text_tokens = 16
    hp.max_speech_tokens = 16
    hp.speech_cond_prompt_len = 0
    return hp


def _make_cond(hp, batch):
    speaker = torch.zeros(batch, hp.speaker_embed_size)
    emotion = torch.zeros(batch, 1, 1)
    return T3Cond(speaker_emb=speaker, emotion_adv=emotion)


class TinyTokenizer:
    def text_to_tokens(self, text, language_id=None):
        count = max(1, len(text.split()))
        tokens = torch.arange(count, dtype=torch.long)
        return tokens.unsqueeze(0)


class TinyS3Gen(torch.nn.Module):
    def __init__(self, mel_lens):
        super().__init__()
        self._mel_lens = mel_lens

    def inference(self, *, speech_tokens, ref_dict, speech_token_lens):
        batch = speech_tokens.size(0)
        wav_len = 8_000
        wavs = torch.stack(
            [torch.arange(wav_len, dtype=torch.float32).unsqueeze(0) for _ in range(batch)]
        ).to(speech_tokens.device)
        mel_tensor = torch.tensor(
            self._mel_lens[:batch], dtype=torch.long, device=speech_tokens.device
        )
        return wavs, mel_tensor


def _make_mtl_tts(mel_lens=(4, 6)):
    hp = _make_tiny_hp()
    t3 = T3(copy.deepcopy(hp))
    s3gen = TinyS3Gen(mel_lens)
    tokenizer = TinyTokenizer()
    ve = object()
    tts = ChatterboxMultilingualTTS(
        t3=t3,
        s3gen=s3gen,
        ve=ve,
        tokenizer=tokenizer,
        device="cpu",
        conds=None,
    )
    speaker_emb = torch.zeros(1, hp.speaker_embed_size)
    emotion_adv = torch.zeros(1, 1, 1)
    conds = MTLConditionals(
        t3=T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_adv),
        gen={"embedding": torch.zeros(1, 1)},
    )
    tts.conds = conds
    return tts


def test_prepare_input_embeds_masks_padded_text_segments():
    hp = _make_tiny_hp()
    t3 = T3(copy.deepcopy(hp))
    batch = 2
    text_tokens = torch.tensor(
        [
            [hp.start_text_token, 1, 2, hp.stop_text_token, hp.stop_text_token, hp.stop_text_token],
            [hp.start_text_token, 3, hp.stop_text_token, hp.stop_text_token, hp.stop_text_token, hp.stop_text_token],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([4, 3], dtype=torch.long)
    speech_tokens = torch.full((batch, 4), hp.start_speech_token, dtype=torch.long)
    cond = _make_cond(hp, batch)

    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=cond,
        text_tokens=text_tokens,
        speech_tokens=speech_tokens,
        text_token_lens=lengths,
    )

    text_region = embeds[:, len_cond : len_cond + text_tokens.size(1), :]
    for idx, valid in enumerate(lengths.tolist()):
        tail = text_region[idx, valid:, :]
        assert torch.allclose(tail, torch.zeros_like(tail), atol=1e-6)


def test_alignment_receives_per_sample_lengths_with_cfg(monkeypatch):
    hp = _make_tiny_hp()
    t3 = T3(copy.deepcopy(hp))
    batch = 2
    text_tokens = torch.tensor(
        [
            [hp.start_text_token, 1, 2, hp.stop_text_token, hp.stop_text_token, hp.stop_text_token],
            [hp.start_text_token, 3, hp.stop_text_token, hp.stop_text_token, hp.stop_text_token, hp.stop_text_token],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([4, 3], dtype=torch.long)
    cond = _make_cond(hp, batch)

    class Recorder:
        instances = []

        def __init__(self, *_, text_tokens_slice, batch_index, eos_idx, **__):
            self.text_tokens_slice = text_tokens_slice
            self.batch_index = batch_index
            self.eos_idx = eos_idx
            self.calls = 0
            Recorder.instances.append(self)

        def step(self, logits, next_token=None):
            self.calls += 1
            return logits

    monkeypatch.setattr("chatterbox.models.t3.t3.AlignmentStreamAnalyzer", Recorder)

    t3.inference(
        t3_cond=cond,
        text_tokens=text_tokens,
        text_token_lens=lengths,
        max_new_tokens=2,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        cfg_weight=0.5,
    )

    assert len(Recorder.instances) == batch
    repeats = 2
    for idx, analyzer in enumerate(Recorder.instances):
        slice_len = analyzer.text_tokens_slice[1] - analyzer.text_tokens_slice[0]
        assert slice_len == lengths[idx].item()
        assert analyzer.batch_index == idx * repeats
        assert analyzer.calls > 0


def test_tail_allowance_limits_post_completion_tokens(monkeypatch):
    hp = _make_tiny_hp()
    hp.alignment_tail_allowance = 2
    t3 = T3(copy.deepcopy(hp))
    batch = 1
    text_tokens = torch.tensor(
        [[hp.start_text_token, 5, hp.stop_text_token, hp.stop_text_token]],
        dtype=torch.long,
    )
    lengths = torch.tensor([3], dtype=torch.long)
    cond = _make_cond(hp, batch)

    class AnalyzerStub:
        def __init__(self, *_, eos_idx=None, **__):
            self.eos_idx = eos_idx
            self.complete = False

        def step(self, logits, next_token=None):
            self.complete = True
            return logits

        class FakeBackend:
            def __init__(self, *_, alignment_stream_analyzer=None, **__):
                self.alignment_stream_analyzer = alignment_stream_analyzer
                self.steps = 0
                self.vocab = t3.hp.speech_tokens_dict_size

        def __call__(
            self,
            inputs_embeds,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        ):
            logits = torch.full(
                    (inputs_embeds.size(0), 1, self.vocab),
                -1e9,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
                token_id = min(10 + self.steps, self.vocab - 1)
            logits[..., token_id] = 0.0
            self.steps += 1
            return types.SimpleNamespace(
                logits=logits,
                past_key_values=tuple(),
            )

    monkeypatch.setattr("chatterbox.models.t3.t3.AlignmentStreamAnalyzer", AnalyzerStub)
    monkeypatch.setattr("chatterbox.models.t3.t3.T3HuggingfaceBackend", FakeBackend)

    outputs = t3.inference(
        t3_cond=cond,
        text_tokens=text_tokens,
        text_token_lens=lengths,
        max_new_tokens=10,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        cfg_weight=0.0,
    )
    assert outputs.shape[1] == hp.alignment_tail_allowance


def test_mtl_generate_batch_trims_each_sample(monkeypatch):
    mel_lengths = (2, 5)
    tts = _make_mtl_tts(mel_lengths)

    def fake_inference(*, text_token_lens, **kwargs):
        lengths = text_token_lens.tolist()
        max_len = max(lengths)
        device = text_token_lens.device
        batch = len(lengths)
        out = torch.zeros(batch, max_len, dtype=torch.long, device=device)
        for idx, L in enumerate(lengths):
            out[idx, :L] = torch.arange(L, dtype=torch.long, device=device)
            out[idx, L:] = tts.t3.hp.stop_speech_token
        return out

    monkeypatch.setattr(tts.t3, "inference", fake_inference)

    outputs = tts.generate_batch(
        ["short text", "a slightly longer sample"],
        language_ids=["en", "en"],
    )

    expected = [m * MTL_SAMPLES_PER_MEL_FRAME for m in mel_lengths]
    actual = [wav.shape[-1] for wav in outputs]
    assert actual == expected


def test_mtl_generate_batch_falls_back_to_speech_len_when_mel_missing(monkeypatch):
    mel_lengths = (0, 4)
    speech_lengths = (3, 5)
    tts = _make_mtl_tts(mel_lengths)

    def fake_inference(*, text_token_lens, **kwargs):
        device = text_token_lens.device
        max_len = max(speech_lengths)
        out = torch.zeros(len(speech_lengths), max_len, dtype=torch.long, device=device)
        for idx, L in enumerate(speech_lengths):
            out[idx, :L] = torch.arange(L, dtype=torch.long, device=device)
            out[idx, L:] = tts.t3.hp.stop_speech_token
        return out

    monkeypatch.setattr(tts.t3, "inference", fake_inference)

    outputs = tts.generate_batch(
        ["alpha beta", "gamma delta epsilon"],
        language_ids=["en", "en"],
    )

    expected = [
        speech_lengths[0] * MTL_SAMPLES_PER_SPEECH_TOKEN,
        mel_lengths[1] * MTL_SAMPLES_PER_MEL_FRAME,
    ]
    actual = [wav.shape[-1] for wav in outputs]
    assert actual == expected
