# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess
from typing import List, Optional
import sys
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

MODEL_PATH = "./checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # Force offline to prevent network downloads and reduce log noise
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        # Download weights
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)

        # Ensure project root (with local radio shim) is importable
        project_root = os.path.abspath(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Load model with GPU support and explicit attention implementation
        # Note: First run may download vision encoder (C-RADIOv2-H) from HuggingFace
        self.model = AutoModel.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            device_map="cuda",
            attn_implementation="eager"  # Use eager attention (most compatible)
        ).eval()
        
        # Load tokenizer from local files
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load image processor from local files
        self.image_processor = AutoImageProcessor.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            device="cuda"
        )

    def predict(
        self,
        image: Path = Input(
            description="Single input image"
        ),
        question: str = Input(
            description="Question or prompt about the image",
            default="Describe the image."
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=1024,
            ge=1,
            le=4096
        ),
        temperature: float = Input(
            description="Temperature for sampling. Set to 0 for greedy decoding.",
            default=0.0,
            ge=0.0,
            le=2.0
        ),
    ) -> str:
        """Run a single prediction on the model"""
        
        # Load and process single image
        img = Image.open(image)
        image_features = self.image_processor(img)
        
        # Generation config
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_config["temperature"] = float(temperature)
        
        # Generate response
        response = self.model.chat(
            tokenizer=self.tokenizer,
            question=question,
            generation_config=generation_config,
            **image_features
        )
        
        # Ensure response is returned as a string
        return str(response)
