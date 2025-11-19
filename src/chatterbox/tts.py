from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"
_MAX_SPEECH_TOKEN_ID = 6561


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
    sentence_enders = {".", "!", "?", "-", ","}
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

        def _tile_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                return t
            if t.size(0) == batch_size:
                return t
            if t.size(0) != 1:
                raise ValueError(
                    f"Cannot expand tensor with batch dimension {t.size(0)} to {batch_size}"
                )
            repeats = [batch_size] + [1] * (t.dim() - 1)
            return t.repeat(*repeats)

        t3_kwargs = {}
        for field, value in self.t3.__dict__.items():
            if torch.is_tensor(value):
                t3_kwargs[field] = _tile_tensor(value)
            else:
                t3_kwargs[field] = value

        expanded_gen = {}
        for key, value in self.gen.items():
            if torch.is_tensor(value):
                expanded_gen[key] = _tile_tensor(value)
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
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
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
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
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
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        return self.generate_batch(
            [text],
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )[0]

    def generate_batch(
        self,
        texts: Sequence[str],
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if not texts:
            raise ValueError("texts must not be empty")

        self._ensure_conditionals(audio_prompt_path, exaggeration)

        text_tokens = self._build_text_batch(texts=texts)
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
        return [wav_batch[i] for i in range(wav_batch.size(0))]

    def _ensure_conditionals(self, audio_prompt_path: str | None, exaggeration: float) -> None:
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        t3_cond = self.conds.t3
        current_adv = t3_cond.emotion_adv
        if torch.is_tensor(current_adv):
            current = float(current_adv.flatten()[0].item())
        else:
            current = float(current_adv)

        if exaggeration != current:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

    def _build_text_batch(self, texts: Sequence[str]) -> torch.LongTensor:
        processed = []
        for raw_text in texts:
            normalized = punc_norm(raw_text)
            tokens = self.tokenizer.text_to_tokens(normalized).squeeze(0).to(self.device)
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
