from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import os

import librosa
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"
_MAX_SPEECH_TOKEN_ID = 6561

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def expand_for_batch(self, batch_size: int) -> tuple[T3Cond, dict]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size == 1:
            return self.t3, self.gen

        def _expand_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                return t
            target_shape = (batch_size, *t.shape[1:])
            return t.expand(*target_shape)

        t3_kwargs = {}
        for field, value in self.t3.__dict__.items():
            if torch.is_tensor(value):
                t3_kwargs[field] = _expand_tensor(value)
            else:
                t3_kwargs[field] = value

        expanded_gen = {}
        for key, value in self.gen.items():
            if torch.is_tensor(value):
                expanded_gen[key] = _expand_tensor(value)
            else:
                expanded_gen[key] = value

        return T3Cond(**t3_kwargs), expanded_gen

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        #self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main", 
                allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        return self.generate_batch(
            [text],
            language_ids=[language_id],
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )[0]


    def generate_batch(
        self,
        texts: Sequence[str],
        language_ids: Sequence[str | None] | None = None,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        if not texts:
            raise ValueError("texts must not be empty")

        if language_ids is None:
            language_ids = [None] * len(texts)

        if len(language_ids) != len(texts):
            raise ValueError("language_ids must match texts length")

        normalized_languages = []
        for lang in language_ids:
            lang_lower = lang.lower() if isinstance(lang, str) else None
            if lang_lower and lang_lower not in SUPPORTED_LANGUAGES:
                supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
                raise ValueError(
                    f"Unsupported language_id '{lang}'. Supported languages: {supported_langs}"
                )
            normalized_languages.append(lang_lower)

        self._ensure_conditionals(audio_prompt_path, exaggeration)
        text_tokens = self._build_text_batch(texts, normalized_languages)
        batch_size = text_tokens.size(0)
        t3_cond_batch, gen_cond_batch = self.conds.expand_for_batch(batch_size)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=t3_cond_batch,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

        speech_tokens, speech_lens = self._prepare_speech_batch(speech_tokens)
        wav_batch, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=gen_cond_batch,
            speech_token_lens=speech_lens,
        )

        wav_batch = wav_batch.detach().cpu()
        results = []
        for i in range(wav_batch.size(0)):
            wav = wav_batch[i].squeeze(0).numpy()
            results.append(torch.from_numpy(wav).unsqueeze(0))
        return results

    def _ensure_conditionals(self, audio_prompt_path: str | None, exaggeration: float) -> None:
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        current = float(self.conds.t3.emotion_adv[0, 0, 0].item())
        if float(exaggeration) != current:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

    def _build_text_batch(
        self, texts: Sequence[str], language_ids: Sequence[str | None]
    ) -> torch.LongTensor:
        processed = []
        for text, lang in zip(texts, language_ids):
            normalized = punc_norm(text)
            tokens = self.tokenizer.text_to_tokens(
                normalized, language_id=lang if lang else None
            ).squeeze(0).to(self.device)
            processed.append(tokens)

        max_len = max(tok.size(0) for tok in processed)
        pad_value = self.t3.hp.stop_text_token
        batch = torch.full((len(processed), max_len), pad_value, dtype=torch.long, device=self.device)
        for idx, tokens in enumerate(processed):
            batch[idx, : tokens.size(0)] = tokens

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        batch = F.pad(batch, (1, 0), value=sot)
        batch = F.pad(batch, (0, 1), value=eot)
        return batch

    def _prepare_speech_batch(
        self, speech_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        trimmed = []
        lengths = []
        for seq in speech_tokens:
            seq = drop_invalid_tokens(seq)
            seq = seq[seq < _MAX_SPEECH_TOKEN_ID]
            if seq.numel() == 0:
                seq = torch.tensor(
                    [self.t3.hp.start_speech_token], dtype=torch.long, device=self.device
                )
            trimmed.append(seq.to(self.device))
            lengths.append(seq.size(0))

        max_len = max(lengths)
        pad_value = 0
        batch = torch.full(
            (len(trimmed), max_len), pad_value, dtype=torch.long, device=self.device
        )
        for idx, seq in enumerate(trimmed):
            batch[idx, : seq.size(0)] = seq

        lens_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        return batch, lens_tensor
