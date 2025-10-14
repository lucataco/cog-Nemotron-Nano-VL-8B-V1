# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import subprocess
from typing import List
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

        # Download weights
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)

        # Ensure local RADIO code is importable for vision backbone
        # radio_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "radio"))
        # if radio_dir not in sys.path:
            # sys.path.insert(0, radio_dir)
        
        # Load model with GPU support and explicit attention implementation
        # Note: First run may download vision encoder (C-RADIOv2-H) from HuggingFace
        # It will be cached in ~/.cache/huggingface for subsequent runs
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
        images: List[Path] = Input(
            description="List of input images. Can be a single image or multiple images for multi-image questions."
        ),
        question: str = Input(
            description="Question or prompt about the image(s)",
            default="Describe the image(s)."
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
        
        # Convert Path objects to PIL Images
        image_list = [Image.open(img_path) for img_path in images]
        
        # Process images
        image_features = self.image_processor(image_list)
        
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
