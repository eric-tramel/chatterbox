import logging
import os
import sys
from pathlib import Path

import torch
import torchaudio

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_batch_inference():
    """
    Test script to verify batched inference fixes.
    Run this on a machine with GPU/MPS support and the model checkpoints.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Running test on device: {device}")

    try:
        logger.info("Loading ChatterboxMultilingualTTS model...")
        # Assuming environment variables or default paths are set up for model loading
        tts = ChatterboxMultilingualTTS.from_pretrained(device)
        
        # Test inputs
        texts = [
            "Hello, this is the first sentence in the batch.",
            "And here is a second, slightly longer sentence to ensure variable lengths work.",
            "Short one.",
            "Finally, a fourth sentence to round out the batch testing."
        ]
        language_id = "en"
        
        logger.info(f"Starting batch generation with batch size {len(texts)}...")
        
        # Perform generation
        # Note: If no internal conditionals are loaded, this might fail without audio_prompt_path.
        # Assuming pretrained model comes with default voice (conds.pt).
        wavs = tts.generate_batch(
            texts=texts,
            language_ids=[language_id] * len(texts),
            # audio_prompt_path="some_ref.wav" # Optional: add if needed
        )
        
        # Verification
        logger.info("Batch generation completed.")
        logger.info(f"Received {len(wavs)} waveforms.")
        
        if len(wavs) != len(texts):
            logger.error(f"Mismatch in output count: expected {len(texts)}, got {len(wavs)}")
            sys.exit(1)
            
        for i, wav in enumerate(wavs):
            if not isinstance(wav, torch.Tensor):
                logger.error(f"Output {i} is not a Tensor")
            else:
                logger.info(f"Waveform {i} shape: {wav.shape}, duration: {wav.shape[-1]/24000:.2f}s")
                if wav.shape[0] != 1:
                     logger.warning(f"Waveform {i} has unexpected channel dim: {wav.shape[0]}")
        
        logger.info("SUCCESS: Batch inference logic executed without errors.")

        # Stitch the batch into a single waveform for manual listening.
        concatenated = torch.cat([wav.squeeze(0) for wav in wavs], dim=0).unsqueeze(0)
        output_dir = Path("artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "batched_generation.wav"
        torchaudio.save(str(output_path), concatenated, tts.sr)
        logger.info(f"Saved concatenated waveform to {output_path.resolve()}")

    except Exception as e:
        logger.error("FAILED: Batch inference encountered an error.", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    test_batch_inference()

