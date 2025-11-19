import copy
import torch
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3 import llama_configs

TINY_LLAMA_CONFIG = dict(
    vocab_size=256,
    max_position_embeddings=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
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
    hp = T3Config.english_only()
    hp.llama_config_name = "TinyTest"
    hp.max_text_tokens = 16
    hp.max_speech_tokens = 16
    hp.speech_cond_prompt_len = 0
    return hp

def _make_cond(hp, batch):
    speaker = torch.zeros(batch, hp.speaker_embed_size)
    emotion = torch.zeros(batch, 1, 1)
    return T3Cond(speaker_emb=speaker, emotion_adv=emotion)

class MockT3HuggingfaceBackend:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def __call__(self, inputs_embeds, position_ids=None, attention_mask=None, **kwargs):
        self.calls.append({
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        })
        # Return mock output
        B, S, D = inputs_embeds.shape
        logits = torch.randn(B, 1, 100) # Fake logits
        
        class MockOutput:
            def __init__(self):
                self.logits = logits
                self.past_key_values = None
        return MockOutput()

def test_position_ids_skip_padding(monkeypatch):
    hp = _make_tiny_hp()
    t3 = T3(copy.deepcopy(hp))
    
    # Capture calls globally
    backend_calls = []
    
    class MockBackend:
        def __init__(self, *args, **kwargs):
            self.calls = backend_calls
            # Needs a dummy config and model to satisfy attribute access if any
            # But mostly it just needs to be callable
            self.config = None

        def __call__(self, inputs_embeds, position_ids=None, attention_mask=None, **kwargs):
            self.calls.append({
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            })
            B, S, D = inputs_embeds.shape
            # Make logits match speech_tokens_dict_size (8194)
            logits = torch.randn(B, 1, 8194)
            class MockOutput:
                def __init__(self):
                    self.logits = logits
                    self.past_key_values = None
            return MockOutput()
            
    monkeypatch.setattr("chatterbox.models.t3.t3.T3HuggingfaceBackend", MockBackend)

    batch = 2
    # Sample 0: Text len 4. Sample 1: Text len 3.
    # Max text len 4.
    # Sample 1 will have 1 pad token at index 3 (0-indexed).
    text_tokens = torch.tensor(
        [
            [hp.start_text_token, 1, 2, 3, hp.stop_text_token], # len 5
            [hp.start_text_token, 1, 2, hp.stop_text_token, hp.stop_text_token], # len 4 (valid), 1 pad
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([5, 4], dtype=torch.long)
    
    cond = _make_cond(hp, batch)
    
    t3.inference(
        t3_cond=cond,
        text_tokens=text_tokens,
        text_token_lens=lengths,
        max_new_tokens=1,
        temperature=1.0,
        cfg_weight=0.0, # Disable CFG for simplicity
    )
    
    call_args = backend_calls[0]
    pos_ids = call_args["position_ids"]
    # pos_ids shape: (B, S)
    # S should be len_cond + len_text + 1 (BOS)
    
    # Check Sample 1 (index 1) which has padding.
    # Expected: [0, 1, ... len_cond-1] (cond) + [len_cond, len_cond+1, len_cond+2, len_cond+3] (text) + [GAP SKIP] + [len_cond+4] (BOS)
    # The GAP is at index len_cond+4 in the sequence.
    # Valid text len is 4. Max text len is 5. 
    # Text tokens are at indices: len_cond, len_cond+1, len_cond+2, len_cond+3.
    # Padding is at index: len_cond+4.
    # BOS is at index: len_cond+5.
    # We want BOS to have position_id = len_cond + 4.
    # And padding at len_cond+4 should ideally have 0 or be skipped.
    
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=cond,
        text_tokens=text_tokens,
        speech_tokens=torch.full((batch, 1), hp.start_speech_token, dtype=torch.long),
        text_token_lens=lengths
    )
    # prepare_input_embeds returns embeds for cond+text+speech.
    # We need len_cond.
    
    print(f"len_cond: {len_cond}")
    
    # Sample 0 (Full length 5)
    # Pos ids should be continuous: 0, 1, ..., len_cond+5
    # Sample 1 (Length 4, 1 pad)
    # Pos ids: 
    #   Cond: 0..len_cond-1
    #   Text: len_cond..len_cond+3
    #   Pad:  (masked/ignored) - we set it to 0 in code, or just shift subsequent?
    #   Code says: 
    #     shift = pad_end - pad_start = (len_cond+5) - (len_cond+4) = 1
    #     position_ids[i, pad_end:] -= shift
    #     position_ids[i, pad_start:pad_end] = 0
    #   So Pad pos should be 0.
    #   BOS pos should be (len_cond+5) - 1 = len_cond+4.
    
    s1_pos = pos_ids[1]
    
    # Check Pad position
    pad_idx = len_cond + 4
    assert s1_pos[pad_idx] == 0, f"Padding position id should be 0, got {s1_pos[pad_idx]}"
    
    # Check BOS position (last token)
    bos_idx = len_cond + 5
    expected_bos_pos = len_cond + 4
    assert s1_pos[bos_idx] == expected_bos_pos, f"BOS position id should be {expected_bos_pos}, got {s1_pos[bos_idx]}"

    # Verify Sample 0 (Full)
    s0_pos = pos_ids[0]
    assert s0_pos[bos_idx] == len_cond + 5, f"Sample 0 BOS position should be {len_cond+5}, got {s0_pos[bos_idx]}"

