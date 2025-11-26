import logging
import os
import sys
import threading
import time
import concurrent.futures
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str) -> ChatterboxMultilingualTTS:
    """Load the TTS model."""
    logger.info(f"Loading ChatterboxMultilingualTTS model on {device}...")
    return ChatterboxMultilingualTTS.from_pretrained(device)


def check_batch_inference(tts: ChatterboxMultilingualTTS) -> List[torch.Tensor]:
    """
    Test basic batched inference.
    
    Returns the generated waveforms for further verification.
    """
    texts = [
        "Hello, this is the first sentence in the batch.",
        "And here is a second, slightly longer sentence to ensure variable lengths work.",
        "Short one.",
        "Finally, a fourth sentence to round out the batch testing."
    ]
    language_id = "en"
    
    logger.info(f"Starting batch generation with batch size {len(texts)}...")
    
    wavs = tts.generate_batch(
        texts=texts,
        language_ids=[language_id] * len(texts),
    )
    
    logger.info("Batch generation completed.")
    logger.info(f"Received {len(wavs)} waveforms.")
    
    if len(wavs) != len(texts):
        raise ValueError(f"Mismatch in output count: expected {len(texts)}, got {len(wavs)}")
        
    for i, wav in enumerate(wavs):
        if not isinstance(wav, torch.Tensor):
            raise TypeError(f"Output {i} is not a Tensor")
        logger.info(f"Waveform {i} shape: {wav.shape}, duration: {wav.shape[-1]/24000:.2f}s")
        if wav.shape[0] != 1:
            logger.warning(f"Waveform {i} has unexpected channel dim: {wav.shape[0]}")
    
    return wavs


def check_sequential_inference(tts: ChatterboxMultilingualTTS) -> None:
    """
    Test multiple sequential inference calls.
    
    This verifies that hooks are properly cleaned up between calls
    and don't accumulate, which was the original bug.
    """
    logger.info("Testing sequential inference calls...")
    
    for i in range(3):
        texts = [f"This is sequential test number {i + 1}.", "Short test."]
        logger.info(f"Sequential call {i + 1}/3...")
        
        wavs = tts.generate_batch(
            texts=texts,
            language_ids=["en"] * len(texts),
        )
        
        assert len(wavs) == len(texts), f"Call {i + 1}: Expected {len(texts)} outputs, got {len(wavs)}"
        
        for j, wav in enumerate(wavs):
            assert isinstance(wav, torch.Tensor), f"Call {i + 1}, output {j} is not a Tensor"
            assert wav.shape[-1] > 0, f"Call {i + 1}, output {j} has zero length"
    
    logger.info("Sequential inference test passed!")


def check_varied_batch_sizes(tts: ChatterboxMultilingualTTS) -> None:
    """
    Test with different batch sizes to ensure robustness.
    """
    logger.info("Testing varied batch sizes...")
    
    test_cases = [
        (1, ["Single item batch test."]),
        (2, ["Two items.", "In this batch."]),
        (5, [f"Item {i} of five." for i in range(5)]),
    ]
    
    for batch_size, texts in test_cases:
        logger.info(f"Testing batch size {batch_size}...")
        
        wavs = tts.generate_batch(
            texts=texts,
            language_ids=["en"] * len(texts),
        )
        
        assert len(wavs) == batch_size, f"Batch size {batch_size}: Expected {batch_size} outputs, got {len(wavs)}"
        
        for i, wav in enumerate(wavs):
            assert isinstance(wav, torch.Tensor)
            assert wav.shape[-1] > 0
    
    logger.info("Varied batch size test passed!")


def check_concurrent_inference(tts: ChatterboxMultilingualTTS, num_threads: int = 4) -> None:
    """
    Test concurrent inference from multiple threads.
    
    This is the key test for thread-safety. Multiple threads will
    attempt to run inference simultaneously, and we verify:
    1. No crashes or exceptions
    2. Each thread gets the correct number of outputs
    3. Outputs are valid audio tensors
    """
    logger.info(f"Testing concurrent inference with {num_threads} threads...")
    
    results: List[Tuple[int, List[torch.Tensor]]] = []
    errors: List[Tuple[int, Exception]] = []
    lock = threading.Lock()
    
    def run_inference(thread_id: int):
        """Run inference in a thread."""
        try:
            texts = [
                f"Thread {thread_id} sentence one.",
                f"Thread {thread_id} sentence two is a bit longer.",
            ]
            
            logger.info(f"Thread {thread_id}: Starting inference...")
            start = time.time()
            
            wavs = tts.generate_batch(
                texts=texts,
                language_ids=["en"] * len(texts),
            )
            
            elapsed = time.time() - start
            logger.info(f"Thread {thread_id}: Completed in {elapsed:.2f}s")
            
            # Verify outputs
            assert len(wavs) == len(texts)
            for wav in wavs:
                assert isinstance(wav, torch.Tensor)
                assert wav.shape[-1] > 0
            
            with lock:
                results.append((thread_id, wavs))
                
        except Exception as e:
            logger.error(f"Thread {thread_id}: Error - {e}")
            with lock:
                errors.append((thread_id, e))
    
    # Run concurrent inferences
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_inference, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)
    
    # Check results
    if errors:
        for thread_id, error in errors:
            logger.error(f"Thread {thread_id} failed: {error}")
        raise RuntimeError(f"{len(errors)} threads encountered errors")
    
    assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
    
    logger.info(f"Concurrent inference test passed! All {num_threads} threads completed successfully.")


def check_hook_cleanup_verification(tts: ChatterboxMultilingualTTS) -> None:
    """
    Verify that the attention hook registry is working correctly.
    
    This test checks:
    1. Registry exists and is initialized
    2. Hooks are registered
    3. After inference, no request contexts are leaked
    """
    logger.info("Testing hook registry state...")
    
    # Access the T3 model's attention registry
    t3 = tts.t3
    
    # Run an inference to initialize the registry
    texts = ["Test sentence for hook verification."]
    tts.generate_batch(texts=texts, language_ids=["en"])
    
    # Check registry state
    assert hasattr(t3, '_attention_registry'), "T3 should have _attention_registry attribute"
    
    registry = t3._attention_registry
    if registry is not None:
        assert registry.is_registered, "Registry should have hooks registered"
        
        # Verify no leaked requests
        with registry._lock:
            num_requests = len(registry._requests)
        
        assert num_requests == 0, f"Expected 0 leaked requests, found {num_requests}"
        logger.info("Hook registry is clean - no leaked requests")
    
    logger.info("Hook cleanup verification passed!")


def save_test_outputs(wavs: List[torch.Tensor], tts: ChatterboxMultilingualTTS, name: str = "batched_generation") -> None:
    """Save test outputs for manual verification."""
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual files
    for i, wav in enumerate(wavs):
        path = output_dir / f"{name}_{i}.wav"
        sf.write(str(path), wav.squeeze(0).cpu().numpy(), tts.sr)
    
    # Save concatenated file
    concatenated = torch.cat([wav.squeeze(0) for wav in wavs], dim=0).unsqueeze(0)
    concat_path = output_dir / f"{name}_concat.wav"
    sf.write(str(concat_path), concatenated.squeeze(0).cpu().numpy(), tts.sr)
    
    logger.info(f"Saved outputs to {output_dir.resolve()}")


def run_all_tests():
    """Run all tests."""
    device = get_device()
    logger.info(f"Running tests on device: {device}")
    
    try:
        tts = load_model(device)
        
        # Test 1: Basic batch inference
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Basic Batch Inference")
        logger.info("="*60)
        wavs = check_batch_inference(tts)
        save_test_outputs(wavs, tts, "test1_batch")
        logger.info("✓ TEST 1 PASSED")
        
        # Test 2: Sequential inference (hook cleanup)
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Sequential Inference (Hook Cleanup)")
        logger.info("="*60)
        check_sequential_inference(tts)
        logger.info("✓ TEST 2 PASSED")
        
        # Test 3: Varied batch sizes
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Varied Batch Sizes")
        logger.info("="*60)
        check_varied_batch_sizes(tts)
        logger.info("✓ TEST 3 PASSED")
        
        # Test 4: Hook cleanup verification
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Hook Registry State")
        logger.info("="*60)
        check_hook_cleanup_verification(tts)
        logger.info("✓ TEST 4 PASSED")
        
        # Test 5: Concurrent inference (thread-safety)
        # Note: This test is more intensive, run with fewer threads on CPU
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Concurrent Inference (Thread-Safety)")
        logger.info("="*60)
        num_threads = 2 if device == "cpu" else 4
        check_concurrent_inference(tts, num_threads=num_threads)
        logger.info("✓ TEST 5 PASSED")
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error("TEST FAILED", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

