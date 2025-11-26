# Copyright (c) 2025 Resemble AI
# MIT License
"""
Thread-safe alignment stream analyzer for concurrent inference.

This module provides a version of AlignmentStreamAnalyzer that works with
the AttentionHookRegistry for safe concurrent use.
"""
import logging
from typing import Optional, Tuple

import torch

from .attention_hook_registry import AttentionHookRegistry, LLAMA_ALIGNED_HEADS

logger = logging.getLogger(__name__)


class ThreadSafeAlignmentStreamAnalyzer:
    """
    Thread-safe version of AlignmentStreamAnalyzer.
    
    Instead of registering its own hooks, this analyzer reads from a shared
    AttentionHookRegistry that manages hooks for all concurrent requests.
    
    Each analyzer instance handles one batch element. The registry captures
    full batch attention, and each analyzer extracts its slice by batch_index.
    """
    
    def __init__(
        self,
        registry: AttentionHookRegistry,
        request_id: str,
        text_tokens_slice: Tuple[int, int],
        eos_idx: int = 0,
        batch_index: int = 0,
        speech_start_idx: Optional[int] = None,
    ):
        """
        Args:
            registry: Shared AttentionHookRegistry instance
            request_id: Unique ID for this inference request (shared by all batch items)
            text_tokens_slice: (start, end) indices for text tokens in attention
            eos_idx: Token ID for end-of-speech
            batch_index: Which batch element this analyzer handles
            speech_start_idx: Where speech tokens start in the sequence
        """
        self.registry = registry
        self.request_id = request_id
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.batch_index = batch_index
        self.speech_start_idx = speech_start_idx if speech_start_idx is not None else j
        
        # Alignment tracking state
        self.alignment = torch.zeros(0, j - i)
        self.curr_frame_pos = 0
        self.text_position = 0
        
        self.started = False
        self.started_at = None
        self.complete = False
        self.completed_at = None
        
        # Token repetition detection
        self.generated_tokens = []
        self.token_repetition_logged = False
        self.force_eos_logged = False
    
    def step(self, logits: torch.Tensor, next_token=None) -> torch.Tensor:
        """
        Analyze alignment and potentially modify logits to force EOS.
        
        This method reads attention from the shared registry, extracting
        only the slice for this analyzer's batch_index.
        
        Args:
            logits: Logits tensor for this batch element (1, vocab_size) or (vocab_size,)
            next_token: The last generated token (for repetition detection)
            
        Returns:
            Potentially modified logits
        """
        # Get attention slice for our batch index from the registry
        attentions = self.registry.get_attention_slice(
            self.request_id, 
            self.batch_index,
            LLAMA_ALIGNED_HEADS
        )
        
        # Validate attention entries - all should have same seq_len
        valid_attns = []
        expected_seq_len = None
        for idx, attn in enumerate(attentions):
            if attn is None:
                continue
            seq_len = attn.shape[-1]
            if expected_seq_len is None:
                expected_seq_len = seq_len
            elif seq_len != expected_seq_len:
                logger.debug(
                    f"Batch {self.batch_index}: attention entry {idx} has seq_len {seq_len}, "
                    f"expected {expected_seq_len}. Skipping."
                )
                continue
            valid_attns.append(attn)
        
        if not valid_attns:
            # No valid attention - can happen on first step before hooks fire
            return logits
        
        # Stack and average attention across heads
        aligned_attn = torch.stack(valid_attns).mean(dim=0)  # (T0, Ti)
        
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # First chunk: skip conditioning and padding to get speech portion
            A_chunk = aligned_attn[self.speech_start_idx:, i:j].clone().cpu()
        else:
            # Subsequent chunks: single frame due to KV-cache
            A_chunk = aligned_attn[:, i:j].clone().cpu()
        
        # Handle edge case of empty chunk
        if A_chunk.numel() == 0:
            self.curr_frame_pos += 1
            return logits
        
        # Apply monotonic masking (prevent attending to future text positions)
        if A_chunk.size(1) > self.curr_frame_pos + 1:
            A_chunk[:, self.curr_frame_pos + 1:] = 0
        
        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)
        A = self.alignment
        T, S = A.shape
        
        # Update text position based on attention
        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn
        
        # Detect false starts (hallucinations at beginning show activations at bottom of attention)
        if T >= 2 and S >= 4:
            false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        else:
            false_start = not self.started
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T
        
        # Check if generation is complete (reached end of text)
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T
        
        # Detect long tail hallucinations (final tokens lasting too long)
        long_tail = False
        if self.complete and self.completed_at is not None and S >= 3:
            tail_sum = A[self.completed_at:, -3:].sum(dim=0)
            if tail_sum.numel() > 0:
                long_tail = tail_sum.max() >= 5
        
        # Detect alignment repetition (activations in earlier tokens after completion)
        alignment_repetition = False
        if self.complete and self.completed_at is not None and S > 5:
            post_complete = A[self.completed_at:, :-5]
            if post_complete.numel() > 0:
                alignment_repetition = post_complete.max(dim=1).values.sum() > 5
        
        # Track generated tokens for repetition detection
        if next_token is not None:
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]
        
        # Check for excessive token repetition (3x same token in a row)
        token_repetition = (
            len(self.generated_tokens) >= 3 and
            len(set(self.generated_tokens[-3:])) == 1
        )
        
        if token_repetition and not self.token_repetition_logged:
            logger.warning(f"🚨 Batch {self.batch_index}: Detected ≥3x repetition of token {self.generated_tokens[-1]}")
            self.token_repetition_logged = True
        
        # Suppress early EOS to prevent premature termination
        if cur_text_posn < S - 3 and S > 5:
            logits[..., self.eos_idx] = -2**15
        
        # Force EOS on bad endings
        if long_tail or alignment_repetition or token_repetition:
            if not self.force_eos_logged:
                logger.warning(
                    f"Batch {self.batch_index}: Forcing EOS - "
                    f"{long_tail=}, {alignment_repetition=}, {token_repetition=}"
                )
                self.force_eos_logged = True
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15
        
        self.curr_frame_pos += 1
        return logits

