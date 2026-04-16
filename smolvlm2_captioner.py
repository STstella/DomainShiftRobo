import torch
import warnings
# [Modified] math and functional have been introduced for subsequent probability and confusion calculations
import math
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .captioner import Captioner

class SmolVLM2Captioner(Captioner):
    """
    A captioner that uses Hugging Face's SmolVLM2-2.2B-Instruct model to generate captions for images.
    """
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct", device='cuda:0'):
        """
        Initialize the SmolVLM2 captioner.
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
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    # _attn_implementation="flash_attention_2"
                ).to(self.device)
            print(f"SmolVLM2Captioner initialized with model: {self.model_id}")
        except Exception as e:
            print(f"Error initializing SmolVLM2 models: {e}")
            raise

    # [Modified] A method specifically designed for calculating and generating text statistical indicators (Logprobs and Perplexity)
    def compute_stats(self, outputs, input_len):
        sequences = outputs.sequences
        scores = outputs.scores
        batch_size = sequences.shape[0]
        gen_len = len(scores)
        all_stats = []

        for batch_idx in range(batch_size):
            # [Modified] Extract the purely generated Token part and exclude the length of the input prompt
            generated_token_ids = sequences[batch_idx, input_len:input_len + gen_len]
            token_logprobs = []

            # [Modified] Traverse each step of the generation and convert logits to logarithmic probabilities through log_softmax
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

    # [Modified] return_stats=False has been added to the parameter list, allowing the caller to choose whether to return statistical metrics
    def caption(self, imgs, user_prompt=None, return_stats=False):
        """
        Caption the given images using the SmolVLM2 model.
        Args:
            imgs: List of images to caption (PIL.Image or numpy arrays)
            user_prompt (str): Custom prompt to use for captioning. 
                               If None, a default caption request will be used.
        Returns:
            List of captions for the images
        """
        if self.processor is None or self.model is None:
            self._init_models()
        if user_prompt is None:
            user_prompt = "Describe this image."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                captions = []
                # [Modified] Initialize the list of statistical information
                all_stats = []
                for img in tqdm(imgs, desc="Captioning images (SmolVLM2)"):
                    try:
                        messages = [
                            {"role": "user", "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": user_prompt}
                            ]}
                        ]
                        inputs = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        ).to(self.model.device, dtype=torch.bfloat16)
                        
                        input_len = inputs["input_ids"].shape[-1]
                        
                        with torch.inference_mode():
                            # [Modified] Replace the original generation variable with outputs, and force the model to return scores and dictionary structure
                            outputs = self.model.generate(
                                **inputs, 
                                max_new_tokens=64, 
                                do_sample=False, 
                                return_dict_in_generate=True, 
                                output_scores=True
                            )
                            
                        # [Modified] Extract the generated ID from the sequences of outputs and decode it. A new.strip() function has been added to remove the first and last whitespace characters
                        generated_ids = outputs.sequences[0][input_len:]
                        caption = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
                        captions.append(caption)

                        # [Modified] If the statistical switch is enabled, calculate various metrics and store them in the list
                        if return_stats:
                            stats = self.compute_stats(outputs, input_len)[0]
                            all_stats.append(stats)

                    except Exception as e:
                        print(f"Error generating caption (SmolVLM2): {e}")
                        captions.append("Error generating caption")
                        # [Modified] If an exception occurs, append an empty dictionary to maintain alignment between caption and statistics lists
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
        print("SmolVLM2Captioner stopped and resources released.")