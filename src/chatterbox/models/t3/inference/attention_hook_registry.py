# Copyright (c) 2025 Resemble AI
# MIT License
"""
Thread-safe attention hook registry for concurrent inference.

This module provides a mechanism to capture attention outputs from transformer
layers in a way that supports multiple concurrent inference requests on the
same model instance.

The design uses:
1. A singleton registry per model that stores attention data keyed by request ID
2. Python's contextvars for async-safe request context tracking
3. Persistent hooks registered once at model init (no accumulation)
4. Full batch attention capture - individual analyzers slice by their batch index
"""
import logging
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# Context variable to track the current request ID in async context
_current_request_id: ContextVar[Optional[str]] = ContextVar('current_request_id', default=None)

# Layer/head pairs used for alignment analysis
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AttentionCapture:
    """
    Captured attention data for a single request (which may be a batch).
    
    Stores full batch attention tensors - individual analyzers slice by batch_index.
    """
    # layer_idx -> full attention tensor (B, n_heads, T0, Ti)
    attentions: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    def get_attention_slice(
        self, 
        layer_head_pairs: List[Tuple[int, int]], 
        batch_index: int
    ) -> List[Optional[torch.Tensor]]:
        """
        Get attention slices for a specific batch index.
        
        Args:
            layer_head_pairs: List of (layer_idx, head_idx) tuples
            batch_index: Which batch element to extract
            
        Returns:
            List of attention tensors, one per layer/head pair
        """
        result = []
        for layer_idx, head_idx in layer_head_pairs:
            attn = self.attentions.get(layer_idx)
            if attn is not None and attn.dim() >= 2:
                # attn shape: (B, n_heads, T0, Ti)
                batch_idx = min(batch_index, attn.size(0) - 1)
                if attn.dim() == 4:
                    # Extract [batch_idx, head_idx] -> (T0, Ti)
                    result.append(attn[batch_idx, head_idx].cpu())
                elif attn.dim() == 3:
                    # Shape (n_heads, T0, Ti) - no batch dim
                    result.append(attn[head_idx].cpu())
                else:
                    result.append(attn.cpu())
            else:
                result.append(None)
        return result
    
    def clear(self) -> None:
        """Clear captured attentions (called between forward passes if needed)."""
        self.attentions.clear()


class AttentionHookRegistry:
    """
    Thread-safe registry for capturing attention outputs across concurrent requests.
    
    Design for batched inference:
    - One request ID per inference call (covers entire batch)
    - Hooks capture full batch attention tensors
    - Individual analyzers (one per batch item) slice by their batch_index
    
    Usage:
        # At model init (once):
        registry = AttentionHookRegistry()
        registry.register_hooks(transformer)
        
        # Per inference call:
        request_id = registry.create_request()
        with registry.request_context(request_id):
            # Run batched inference - hooks capture full batch attention
            output = model(batch_input)
            
            # Each analyzer gets its slice
            for i, analyzer in enumerate(analyzers):
                attns = registry.get_attention_slice(request_id, batch_index=i)
                analyzer.step(logits[i], attns)
        
        registry.cleanup_request(request_id)
    """
    
    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._requests: Dict[str, AttentionCapture] = {}
        self._hook_handles: List = []
        self._registered = False
    
    @property
    def is_registered(self) -> bool:
        """Check if hooks have been registered."""
        return self._registered
    
    def register_hooks(self, tfmr, layer_indices: Optional[List[int]] = None) -> None:
        """
        Register attention capture hooks on specified layers.
        Should be called ONCE during model initialization.
        
        Args:
            tfmr: The transformer model (e.g., LlamaModel)
            layer_indices: Which layers to hook. Defaults to layers from LLAMA_ALIGNED_HEADS.
        """
        with self._lock:
            if self._registered:
                logger.debug("Hooks already registered, skipping")
                return
            
            if layer_indices is None:
                # Extract unique layer indices from aligned heads config
                layer_indices = list(set(layer_idx for layer_idx, _ in LLAMA_ALIGNED_HEADS))
            
            # Get the number of layers in the model
            num_layers = len(tfmr.layers) if hasattr(tfmr, 'layers') else 0
            
            registered_layers = []
            for layer_idx in layer_indices:
                # Skip layers that don't exist in this model (e.g., tiny test models)
                if layer_idx >= num_layers:
                    logger.debug(f"Skipping layer {layer_idx} (model only has {num_layers} layers)")
                    continue
                handle = self._register_layer_hook(tfmr, layer_idx)
                self._hook_handles.append(handle)
                registered_layers.append(layer_idx)
            
            # Enable attention output on the transformer config
            if hasattr(tfmr, 'config'):
                tfmr.config.output_attentions = True
            
            self._registered = True
            if registered_layers:
                logger.debug(f"Registered attention hooks on layers {registered_layers}")
            else:
                logger.debug("No layers available for attention hooks")
    
    def _register_layer_hook(self, tfmr, layer_idx: int):
        """Register a forward hook on a specific attention layer."""
        def attention_forward_hook(module, input, output):
            request_id = _current_request_id.get()
            if request_id is None:
                return  # No active request context, skip capture
            
            with self._lock:
                capture = self._requests.get(request_id)
                if capture is None:
                    return  # Request not registered or already cleaned up
                
                # Capture attention weights if available
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    # Store full batch attention (don't move to CPU yet for efficiency)
                    # Shape: (B, n_heads, T0, Ti)
                    capture.attentions[layer_idx] = output[1].detach()
        
        target_layer = tfmr.layers[layer_idx].self_attn
        return target_layer.register_forward_hook(attention_forward_hook)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks. Call during model cleanup."""
        with self._lock:
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()
            self._registered = False
    
    def create_request(self) -> str:
        """
        Create a new request context and return its ID.
        
        Call this at the start of each inference call.
        """
        request_id = str(uuid.uuid4())
        with self._lock:
            self._requests[request_id] = AttentionCapture()
        return request_id
    
    def cleanup_request(self, request_id: str) -> None:
        """
        Clean up a request's captured data.
        
        Call this at the end of each inference call.
        """
        with self._lock:
            capture = self._requests.pop(request_id, None)
            if capture:
                capture.clear()
    
    def get_attention_slice(
        self, 
        request_id: str, 
        batch_index: int,
        layer_head_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Optional[torch.Tensor]]:
        """
        Get captured attention slices for a specific batch element.
        
        Args:
            request_id: The request ID from create_request()
            batch_index: Which batch element to extract
            layer_head_pairs: Which layer/head pairs to get. Defaults to LLAMA_ALIGNED_HEADS.
            
        Returns:
            List of attention tensors (T0, Ti), one per layer/head pair
        """
        if layer_head_pairs is None:
            layer_head_pairs = LLAMA_ALIGNED_HEADS
            
        with self._lock:
            capture = self._requests.get(request_id)
            if capture is None:
                return [None] * len(layer_head_pairs)
            return capture.get_attention_slice(layer_head_pairs, batch_index)
    
    @contextmanager
    def request_context(self, request_id: str):
        """
        Context manager to set the current request ID for attention capture.
        
        All forward passes within this context will have their attention
        captured to the specified request.
        """
        token = _current_request_id.set(request_id)
        try:
            yield
        finally:
            _current_request_id.reset(token)

