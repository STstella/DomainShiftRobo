import ollama
# [Modified] math is introduced for calculating Perplexity
import math
# [Modified] Introduce io for image byte stream conversion
import io
from tqdm import tqdm
import tempfile
import subprocess as sp
from .captioner import Captioner
from PIL import Image

class OllamaCaptioner(Captioner):
    """
    A captioner that uses Ollama to generate captions for images.
    """
    def __init__(self, model_name="gemma3:12b", device=None):
        """
        Initialize the Ollama captioner.
        
        Args:
            model_name (str): Name of the Ollama model to use
            device (str): Device is ignored as Ollama handles its own device assignment
        """
        self.model_name = model_name
        super().__init__(device)  # device is ignored for Ollama
        print(f"OllamaCaptioner initialized with model: {self.model_name}")

    def _init_models(self):
        pass  # Ollama models are initialized on demand

    def caption(self, imgs, user_prompt=None, return_stats=False):
        """
        Caption the given images using the Ollama model.
        
        Args:
            imgs: List of images to caption
            user_prompt (str): Custom prompt to use for captioning
            
        Returns:
            List of captions for the images
        """
        captions = []
        # [Modified] Initialize an empty list to store the statistical metrics for each image
        all_stats = []
        print(f'Captioning images with Ollama model ({self.model_name})...')

        with tqdm(total=len(imgs)) as pbar:
            for img in imgs:
                try:
                    # Convert PIL Image to bytes
                    import io 
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_byte_arr = img_byte_arr.getvalue()

                    messages=[{
                            'role': 'user',
                            'content': user_prompt,
                            'images': [img_byte_arr]
                    }]

                    # Use the ollama Python package to send the request
                    # [Modified] When sending a request, require the Ollama engine to return statistical parameters such as logprobs
                    response = ollama.chat(
                        model=self.model_name,
                        messages=messages,
                        options={
                            "num_predict": 20
                        },
                        stream=False,       # [Modified] Make sure not to use streaming output
                        logprobs=True,      # [Modified] Enable logprobs to obtain probabilities
                        top_logprobs=3      # [Modified] Get the top 3 candidate probabilities
                    )

                    # Extract the response content
                    caption = response['message']['content']

                    # [Modified] If statistical information is required to be returned, parse the logarithmic probability returned by Ollama and calculate the confusion
                    if return_stats:
                        logprob_items = response.get("logprobs", [])
                        token_logprobs = [item["logprob"] for item in logprob_items if "logprob" in item]

                        if len(token_logprobs) > 0:
                            avg_logprob = sum(token_logprobs) / len(token_logprobs)
                            perplexity = math.exp(-avg_logprob)
                        else:
                            avg_logprob = None
                            perplexity = None

                        all_stats.append({
                            "token_logprobs": token_logprobs if len(token_logprobs) > 0 else None,
                            "avg_logprob": avg_logprob,
                            "perplexity": perplexity
                        })

                except Exception as e:
                    print(f"Error generating caption: {e}")
                    caption = "ERROR"
                    if return_stats:
                        all_stats.append({
                            "token_logprobs": None,
                            "avg_logprob": None,
                            "perplexity": None
                            })

                captions.append(caption)
                pbar.update(1)
        
        
        if return_stats:
            return captions, all_stats

        return captions

    def stop(self):
        """
        Stop the captioner and release any resources.
        """
        print("Stopping OllamaCaptioner...")
        sp.Popen(["ollama", "stop", self.model_name])