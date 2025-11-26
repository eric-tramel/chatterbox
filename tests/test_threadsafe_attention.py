# Copyright (c) 2025 Resemble AI
# MIT License
"""
Unit tests for thread-safe attention hook infrastructure.

These tests verify:
1. AttentionHookRegistry basic functionality
2. Request isolation (concurrent requests don't interfere)
3. Context variable propagation
4. Cleanup behavior
5. ThreadSafeAlignmentStreamAnalyzer attention handling

Run with: pytest tests/test_threadsafe_attention.py -v
"""
import concurrent.futures
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from chatterbox.models.t3.inference.attention_hook_registry import (
    AttentionHookRegistry,
    LLAMA_ALIGNED_HEADS,
    _current_request_id,
)
from chatterbox.models.t3.inference.threadsafe_alignment_analyzer import (
    ThreadSafeAlignmentStreamAnalyzer,
)


class MockSelfAttention(nn.Module):
    """Mock attention module for testing hooks."""
    def __init__(self, num_heads=16):
        super().__init__()
        self._hooks = []
        self.num_heads = num_heads
    
    def forward(self, x, output_attentions=False):
        # Simulate attention output: (output, attention_weights, past_kv)
        B, S, D = x.shape
        attn_weights = torch.rand(B, self.num_heads, S, S)
        return x, attn_weights, None


class MockTransformerLayer(nn.Module):
    """Mock transformer layer with self_attn."""
    def __init__(self, num_heads=16):
        super().__init__()
        self.self_attn = MockSelfAttention(num_heads=num_heads)


class MockTransformer(nn.Module):
    """Mock transformer with multiple layers for testing."""
    def __init__(self, num_layers=16, num_heads=16):
        super().__init__()
        self.layers = nn.ModuleList([MockTransformerLayer(num_heads=num_heads) for _ in range(num_layers)])
        self.config = MagicMock()
        self.config.output_attentions = False


class TestAttentionHookRegistry:
    """Tests for AttentionHookRegistry."""
    
    def test_create_and_cleanup_request(self):
        """Test basic request lifecycle."""
        registry = AttentionHookRegistry()
        
        # Create request
        request_id = registry.create_request()
        assert request_id is not None
        assert len(request_id) > 0
        
        # Request should exist
        with registry._lock:
            assert request_id in registry._requests
        
        # Cleanup
        registry.cleanup_request(request_id)
        
        # Request should be gone
        with registry._lock:
            assert request_id not in registry._requests
    
    def test_register_hooks_once(self):
        """Test that hooks are only registered once."""
        registry = AttentionHookRegistry()
        tfmr = MockTransformer()
        
        # First registration
        registry.register_hooks(tfmr)
        assert registry.is_registered
        num_hooks = len(registry._hook_handles)
        assert num_hooks > 0
        
        # Second registration should be no-op
        registry.register_hooks(tfmr)
        assert len(registry._hook_handles) == num_hooks
        
        # Cleanup
        registry.remove_hooks()
        assert not registry.is_registered
        assert len(registry._hook_handles) == 0
    
    def test_request_context_sets_context_var(self):
        """Test that request_context properly sets the context variable."""
        registry = AttentionHookRegistry()
        request_id = registry.create_request()
        
        # Outside context
        assert _current_request_id.get() is None
        
        # Inside context
        with registry.request_context(request_id):
            assert _current_request_id.get() == request_id
        
        # After context
        assert _current_request_id.get() is None
        
        registry.cleanup_request(request_id)
    
    def test_request_context_exception_safety(self):
        """Test that context is cleaned up even on exceptions."""
        registry = AttentionHookRegistry()
        request_id = registry.create_request()
        
        try:
            with registry.request_context(request_id):
                assert _current_request_id.get() == request_id
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Context should be reset even after exception
        assert _current_request_id.get() is None
        registry.cleanup_request(request_id)
    
    def test_hook_captures_attention(self):
        """Test that hooks capture attention when context is set."""
        registry = AttentionHookRegistry()
        tfmr = MockTransformer(num_layers=16, num_heads=16)
        
        # Use simple layer indices for testing
        test_layers = [0, 1, 2]
        test_layer_head_pairs = [(0, 0), (1, 1), (2, 2)]
        registry.register_hooks(tfmr, layer_indices=test_layers)
        
        request_id = registry.create_request()
        
        with registry.request_context(request_id):
            # Simulate forward pass - manually trigger hooks
            x = torch.rand(2, 10, 64)  # batch=2, seq=10, dim=64
            for layer_idx in test_layers:
                output = tfmr.layers[layer_idx].self_attn(x, output_attentions=True)
                # Manually fire hooks (in real usage, PyTorch does this)
                for hook in tfmr.layers[layer_idx].self_attn._forward_hooks.values():
                    hook(tfmr.layers[layer_idx].self_attn, (x,), output)
        
        # Check attention was captured using our test pairs
        attentions = registry.get_attention_slice(
            request_id, batch_index=0, layer_head_pairs=test_layer_head_pairs
        )
        assert len(attentions) == len(test_layer_head_pairs)
        # All should be captured since we used matching layers
        for attn in attentions:
            assert attn is not None
        
        registry.cleanup_request(request_id)
        registry.remove_hooks()
    
    def test_request_isolation(self):
        """Test that concurrent requests don't interfere with each other."""
        registry = AttentionHookRegistry()
        
        request1 = registry.create_request()
        request2 = registry.create_request()
        
        # Simulate storing different data
        with registry._lock:
            registry._requests[request1].attentions[9] = torch.ones(2, 8, 10, 10)
            registry._requests[request2].attentions[9] = torch.zeros(2, 8, 10, 10)
        
        # Verify isolation
        attn1 = registry.get_attention_slice(request1, batch_index=0, layer_head_pairs=[(9, 0)])
        attn2 = registry.get_attention_slice(request2, batch_index=0, layer_head_pairs=[(9, 0)])
        
        assert attn1[0] is not None
        assert attn2[0] is not None
        assert torch.all(attn1[0] == 1.0)
        assert torch.all(attn2[0] == 0.0)
        
        registry.cleanup_request(request1)
        registry.cleanup_request(request2)
    
    def test_concurrent_request_creation(self):
        """Test thread-safety of concurrent request creation."""
        registry = AttentionHookRegistry()
        num_threads = 10
        request_ids = []
        lock = threading.Lock()
        
        def create_request():
            req_id = registry.create_request()
            with lock:
                request_ids.append(req_id)
            time.sleep(0.01)  # Simulate some work
            return req_id
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_request) for _ in range(num_threads)]
            concurrent.futures.wait(futures)
        
        # All requests should be unique
        assert len(request_ids) == num_threads
        assert len(set(request_ids)) == num_threads
        
        # All requests should exist in registry
        with registry._lock:
            for req_id in request_ids:
                assert req_id in registry._requests
        
        # Cleanup
        for req_id in request_ids:
            registry.cleanup_request(req_id)
    
    def test_batch_index_slicing(self):
        """Test that attention is correctly sliced by batch index."""
        registry = AttentionHookRegistry()
        request_id = registry.create_request()
        
        # Simulate captured attention with different values per batch
        batch_size = 4
        seq_len = 20
        num_heads = 8
        attn = torch.arange(batch_size).float().view(batch_size, 1, 1, 1)
        attn = attn.expand(batch_size, num_heads, seq_len, seq_len)
        
        with registry._lock:
            registry._requests[request_id].attentions[9] = attn
        
        # Verify each batch index gets correct slice
        for batch_idx in range(batch_size):
            slices = registry.get_attention_slice(
                request_id, 
                batch_index=batch_idx,
                layer_head_pairs=[(9, 0)]
            )
            assert slices[0] is not None
            # Each slice should have the batch index value
            assert torch.allclose(slices[0], torch.full((seq_len, seq_len), float(batch_idx)))
        
        registry.cleanup_request(request_id)


class TestThreadSafeAlignmentStreamAnalyzer:
    """Tests for ThreadSafeAlignmentStreamAnalyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        registry = AttentionHookRegistry()
        request_id = registry.create_request()
        
        analyzer = ThreadSafeAlignmentStreamAnalyzer(
            registry=registry,
            request_id=request_id,
            text_tokens_slice=(10, 50),
            eos_idx=1,
            batch_index=0,
            speech_start_idx=60,
        )
        
        assert analyzer.text_tokens_slice == (10, 50)
        assert analyzer.eos_idx == 1
        assert analyzer.batch_index == 0
        assert analyzer.speech_start_idx == 60
        assert analyzer.curr_frame_pos == 0
        assert not analyzer.complete
        assert not analyzer.started
        
        registry.cleanup_request(request_id)
    
    def test_analyzer_step_with_no_attention(self):
        """Test that step returns logits unchanged when no attention available."""
        registry = AttentionHookRegistry()
        request_id = registry.create_request()
        
        analyzer = ThreadSafeAlignmentStreamAnalyzer(
            registry=registry,
            request_id=request_id,
            text_tokens_slice=(10, 50),
            eos_idx=1,
            batch_index=0,
        )
        
        logits = torch.randn(1, 100)
        original_logits = logits.clone()
        
        result = analyzer.step(logits)
        
        # Should return unchanged when no attention
        assert torch.equal(result, original_logits)
        
        registry.cleanup_request(request_id)
    
    def test_analyzer_step_with_valid_attention(self):
        """Test that step processes logits when attention is available."""
        registry = AttentionHookRegistry()
        request_id = registry.create_request()
        
        # Set up attention data
        seq_len = 100
        text_start, text_end = 10, 50
        speech_start = 60
        num_heads = 16
        
        # Create mock attention that looks reasonable
        # Shape: (B, n_heads, T, T)
        attn = torch.rand(1, num_heads, seq_len, seq_len)
        # Make diagonal stronger (simulating alignment)
        for i in range(seq_len):
            attn[0, :, i, max(0, i-5):min(seq_len, i+5)] *= 2
        
        with registry._lock:
            for layer_idx, _ in LLAMA_ALIGNED_HEADS:
                registry._requests[request_id].attentions[layer_idx] = attn.clone()
        
        analyzer = ThreadSafeAlignmentStreamAnalyzer(
            registry=registry,
            request_id=request_id,
            text_tokens_slice=(text_start, text_end),
            eos_idx=1,
            batch_index=0,
            speech_start_idx=speech_start,
        )
        
        logits = torch.randn(1, 100)
        result = analyzer.step(logits)
        
        # Verify analyzer state updated
        assert analyzer.curr_frame_pos == 1
        
        registry.cleanup_request(request_id)
    
    def test_analyzer_token_repetition_detection(self):
        """Test that analyzer detects token repetition."""
        registry = AttentionHookRegistry()
        request_id = registry.create_request()
        
        analyzer = ThreadSafeAlignmentStreamAnalyzer(
            registry=registry,
            request_id=request_id,
            text_tokens_slice=(10, 50),
            eos_idx=1,
            batch_index=0,
        )
        
        # Simulate repeated tokens
        analyzer.generated_tokens = [42, 42, 42]
        
        # Check repetition detection
        assert len(set(analyzer.generated_tokens[-3:])) == 1
        
        registry.cleanup_request(request_id)
    
    def test_multiple_analyzers_same_request(self):
        """Test multiple analyzers for batched inference."""
        registry = AttentionHookRegistry()
        request_id = registry.create_request()
        
        # Create analyzers for batch of 4
        analyzers = []
        for batch_idx in range(4):
            analyzer = ThreadSafeAlignmentStreamAnalyzer(
                registry=registry,
                request_id=request_id,
                text_tokens_slice=(10, 50),
                eos_idx=1,
                batch_index=batch_idx,
            )
            analyzers.append(analyzer)
        
        # Verify all have same request_id but different batch_index
        for i, analyzer in enumerate(analyzers):
            assert analyzer.request_id == request_id
            assert analyzer.batch_index == i
        
        registry.cleanup_request(request_id)


class TestConcurrentInference:
    """Tests simulating concurrent inference scenarios."""
    
    def test_concurrent_contexts_isolated(self):
        """Test that concurrent contexts are properly isolated."""
        registry = AttentionHookRegistry()
        results = {}
        errors = []
        
        def simulate_inference(thread_id):
            try:
                request_id = registry.create_request()
                
                # Store unique data for this request
                with registry._lock:
                    registry._requests[request_id].attentions[9] = torch.full(
                        (1, 8, 10, 10), float(thread_id)
                    )
                
                with registry.request_context(request_id):
                    # Verify context is set correctly
                    assert _current_request_id.get() == request_id
                    
                    # Simulate some work
                    time.sleep(0.01)
                    
                    # Verify our data is still there
                    slices = registry.get_attention_slice(
                        request_id, batch_index=0, layer_head_pairs=[(9, 0)]
                    )
                    
                    results[thread_id] = slices[0][0, 0].item()
                
                registry.cleanup_request(request_id)
            except Exception as e:
                errors.append((thread_id, e))
        
        # Run concurrent inferences
        num_threads = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(simulate_inference, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify each thread got its own data
        assert len(results) == num_threads
        for thread_id, value in results.items():
            assert value == float(thread_id), f"Thread {thread_id} got wrong value: {value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

