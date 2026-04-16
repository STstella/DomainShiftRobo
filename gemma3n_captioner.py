import torch
# [Modified] Introduce functional for calculating log_softmax
import torch.nn.functional as F
# [Modified] Introduce math for calculating confusion (math.exp)
import math
import warnings
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from .captioner import Captioner

class Gemma3nCaptioner(Captioner):
    """
    A captioner that uses Hugging Face's Gemma-3n model to generate captions for images.
    """
    def __init__(self, model_id="google/gemma-3n-e4b-it", device='cuda:0'):
        """
        Initialize the HF captioner.
        
        Args:
            model_id (str): Model ID from Hugging Face model hub
            device (str): Device to use for model inference
        """
        self.model_id = model_id
        self.processor = None
        self.model = None
        super().__init__(device)
        self._init_models()

    def _init_models(self):
        """Initialize the processor and model"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = Gemma3nForConditionalGeneration.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                ).eval()

            print(f"HFCaptioner initialized with model: {self.model_id}")
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise
            
    # [Modified] Extract and calculate the average log probability and perplexity of the generated content
    def compute_stats(self, outputs, input_len):
        sequences = outputs.sequences
        scores = outputs.scores
        batch_size = sequences.shape[0]
        gen_len = len(scores)
        all_stats = []

        for batch_idx in range(batch_size):
            # [Modified] Extract purely generated tokens and exclude the part where the prompt is input (input_len)
            generated_token_ids = sequences[batch_idx, input_len:input_len + gen_len]
            token_logprobs = []
            
            # [Modified] Traverse each generated token and extract the corresponding log probability
            for step_idx, step_scores in enumerate(scores):
                log_probs = F.log_softmax(step_scores[batch_idx], dim=-1)
                token_id = generated_token_ids[step_idx].item()
                token_logprob = log_probs[token_id].item()
                token_logprobs.append(token_logprob)
                
            if len(token_logprobs) > 0:
                avg_logprob = sum(token_logprobs) / len(token_logprobs)
                perplexity = math.exp(-avg_logprob)
            else:
                avg_logprob = None
                perplexity = None

            all_stats.append({
                "token_logprobs": token_logprobs,
                "avg_logprob": avg_logprob,
                "perplexity": perplexity
            })
        return all_stats

    # [Modified] Function signature adds return_stats=False control flag
    def caption(self, imgs, user_prompt, return_stats=False):
        """
        Caption the given images using the Hugging Face model.
        
        Args:
            imgs: List of images to caption
            user_prompt (str): Custom prompt to use for captioning. 
                               If None, a default caption request will be used.
                
        Returns:
            List of captions for the images
        """
        # Initialize models if they haven't been initialized
        if self.processor is None or self.model is None:
            self._init_models()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                captions = []
                # [Modified] Initialize an empty list to store statistics for each image
                all_stats = []

                for img in tqdm(imgs, desc="Captioning images"):
                    try:
                        messages = [
                            {"role": "user", "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": user_prompt}
                            ]}
                        ]

                        # Process inputs
                        inputs = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        ).to(self.model.device)

                        input_len = inputs["input_ids"].shape[-1]

                        # [Modified] Originally, the text ID was directly generated. Now, it is required to return the dictionary structure and output the scores probability distribution
                        with torch.inference_mode():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=1000,
                                do_sample=False,
                                return_dict_in_generate=True,
                                output_scores=True
                            )

                        # [Modified] Slice from outputs.sequences to obtain the generated text ID and remove the Spaces at both ends
                        generated_ids = outputs.sequences[0][input_len:]
                        caption = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
                        captions.append(caption)

                        # [Modified] If you need to return statistical information, call the newly added compute_stats to obtain it and store it in the list
                        if return_stats:
                            stats = self.compute_stats(outputs, input_len)[0]
                            all_stats.append(stats)

                        # Generate caption 
                        # with torch.inference_mode():
                        #    generation = self.model.generate(**inputs, max_new_tokens=1000, do_sample=False)
                        #   generation = generation[0][input_len:]

                        # caption = self.processor.decode(generation, skip_special_tokens=True)
                        # captions.append(caption)

                    except Exception as e:
                        print(f"Error generating caption: {e}")
                        captions.append("Error generating caption")
                        if return_stats:
                            all_stats.append({
                                "token_logprobs": None,
                                "avg_logprob": None,
                                "perplexity": None
                            })

                if return_stats:
                    return captions, all_stats

                return captions

    def stop(self):
        """Stop the captioner and release resources"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        print("HFCaptioner stopped and resources released.")