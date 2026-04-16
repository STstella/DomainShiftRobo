from captioning.blip2_captioner import BLIP2Captioner
from captioning.ollama_captioner import OllamaCaptioner
from captioning.gemma3n_captioner import Gemma3nCaptioner
from captioning.smolvlm2_captioner import SmolVLM2Captioner
# from captioning.moondream_captioner import MoondreamCaptioner

import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd

# VLMs = ["llama_vision", "qwen", "gemma", "llava", "gemma3n", "mistral", "smolvlm2", "moondream"]
VLMs = ["llama_vision", "qwen", "gemma", "llava", "gemma3n", "mistral", "smolvlm2"]


def parse_args():
    parser = argparse.ArgumentParser(description="Image Captioning with Segmentor and Captioner")
    parser.add_argument("--config", type=str, default="configs.json", help="Path to the config file")
    parser.add_argument("--captioner", type=str, default="blip2", help="Captioner model to use")
    parser.add_argument("--use_masks", action="store_true", help="Use masks for captioning")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name for captioning")
    parser.add_argument("--tot_vram_gb", type=int, default=None, help="Total VRAM in GB")
    return parser.parse_args()

def load_config(config_path):
    """Load the configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

def load_seg_results(tracking_data_path):
    """Load segmentation results from a cached file."""
    tracking_data = np.load(tracking_data_path, allow_pickle=True)
    frame_count = tracking_data["frame_count"].item()
    frame_indices = tracking_data["frame_indices"].tolist()
    frame_data = {}
    print("Loading cached segmentation results...")
    for idx in tqdm(frame_indices):
        data = tracking_data[f"frame_{idx}"].item()
        num_objects = data["num_objects"]
        masks = data["masks"]
        # Convert masks to a list of numpy arrays
        # Make the mask binary (0 or 1)
        masks = [(mask > 0).astype(np.uint8) for mask in masks]
        bounding_boxes = data["bboxes"]
        # Convert bounding boxes to a list of tuples
        bounding_boxes = [tuple(box) for box in bounding_boxes]
        # Store the data in the dictionary
        frame_data[idx] = (num_objects, masks, bounding_boxes)

    return frame_count, frame_data

def select_captioner(captioner_name, tot_vram_gb, device):
    """Select the captioner based on the name."""
    if captioner_name == "blip2":
        return BLIP2Captioner(device=device)
    elif captioner_name == "gemma3n":
        return Gemma3nCaptioner(model_id="google/gemma-3n-e4b-it", device=device)
    elif captioner_name == "smolvlm2":
        return SmolVLM2Captioner(device=device)
    # elif captioner_name == "moondream":
    #     return MoondreamCaptioner(device=device)
    elif captioner_name in VLMs:
        ollama_map_path = os.path.join(os.path.dirname(__file__), "ollama_map.json")
        with open(ollama_map_path, "r") as f:
            ollama_map = json.load(f)
        tot_vram_gb = str(tot_vram_gb)  # Convert to string for lookup
        if captioner_name in VLMs:
            model_name = ollama_map[captioner_name][tot_vram_gb]  # Default to 12B model if no VRAM info            
        else:
            raise ValueError(f"Unknown VLM model: {captioner_name}")
        return OllamaCaptioner(model_name=model_name, device=device)
    else:
        raise ValueError(f"Unknown captioner model: {captioner_name}")

def crop_and_mask(image, bounding_boxes, masks=None):
    """Preprocess the image for captioning."""
    min_size = 128
    empty_images_ids = []
    cropped_images = []
    for i, box in enumerate(bounding_boxes):
        if masks is not None:
            # Apply the mask to the image
            mask = masks[i]
            # Convert mask to PIL Image
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
            # Create a new image with the same size as the original
            new_image = Image.new("RGB", image.size)
            # Paste the original image into the new image using the mask
            new_image.paste(image, mask=mask_image)
        else:
            # If no masks are provided, use the original image
            new_image = image
        # Crop the image based on the bounding box
        # Bounding boxes are in XYWH format (x, y, width, height)
        x, y, w, h = box
        x1, y1, x2, y2 = x, y, x + w, y + h
        cropped_image = new_image.crop((x1, y1, x2, y2))
        # Check if the cropped image is empty
        if cropped_image.getbbox() is None:
            empty_images_ids.append(i)
            cropped_image = Image.new("RGB", size=(min_size, min_size))  # Create a dummy image

        # Ensure the one of the dimension of the cropped image is at least 128
        # otherwise resize it so that at least one dimension is 128 and the other is scaled accordingly
        # This is to avoid issues with some captioners that require a minimum size
        width, height = cropped_image.size
        if width < min_size or height < min_size:
            if width < height:
                new_width = min_size
                new_height = int((min_size / width) * height)
            else:
                new_height = min_size
                new_width = int((min_size / height) * width)
            cropped_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)

        cropped_images.append(cropped_image)
    return cropped_images, empty_images_ids

def generate_captions(captioner_name, captioner, image, bounding_boxes, masks=None, user_prompt=None):
    """Generate captions for the given image and bounding boxes."""

    # Crop and mask the image based on the bounding boxes
    cropped_images, empty_images_ids = crop_and_mask(image, bounding_boxes, masks)

    # [Modified] All captioner.caption calls have added the return_stats=True parameter
    # [Modified] And use the two variables captions and caption_stats to receive the return value
    if captioner_name == "blip2":
        captions, caption_stats = captioner.caption(cropped_images, return_stats=True)
    elif captioner_name == "gemma3n":
        captions, caption_stats = captioner.caption(cropped_images, user_prompt=user_prompt, return_stats=True)
    elif captioner_name == "smolvlm2":
        captions, caption_stats = captioner.caption(cropped_images, user_prompt=user_prompt, return_stats=True)
    elif captioner_name in VLMs:
        captions, caption_stats = captioner.caption(
            cropped_images,
            user_prompt=user_prompt,
            return_stats=True
        )
        # Generate caption for the cropped images with a user prompt
        # captions = captioner.caption(cropped_images, user_prompt=user_prompt)
        # caption_stats = [
        #     {"token_logprobs": None, "avg_logprob": None, "perplexity": None}
        #     for _ in captions
        # ]
    else:
        # Generate caption for the cropped images without a user prompt
        captions = captioner.caption(cropped_images)
        caption_stats = [
            {"token_logprobs": None, "avg_logprob": None, "perplexity": None}
            for _ in captions
        ]

    # Replace captions of empty images with a placeholder
    for i in empty_images_ids:
        captions[i] = "???"
        caption_stats[i] = {
            "token_logprobs": None,
            "avg_logprob": None,
            "perplexity": None
        }

    return captions, caption_stats


def caption():
    args = parse_args()
    config = load_config(args.config)
    dataset = args.dataset
    if args.captioner in VLMs:
        user_prompt_vlm = config["prompts"]["vlm"][args.captioner]
        # print("User prompt for VLM:", user_prompt_vlm)
    elif args.captioner == "blip2":
        user_prompt_vlm = None
    else:
        raise ValueError(f"Unknown VLM model: {args.captioner}")
    config = config["general"]

    # Load cached segmentation results, which where saves in a .npz file

    tracking_data_path = os.path.join(f"output/{dataset}/sam2_tracking", "tracking_data.npz")
    if not os.path.exists(tracking_data_path):
        raise FileNotFoundError(f"Tracking data file {tracking_data_path} not found.")
    frame_count, frame_data = load_seg_results(tracking_data_path)

    # frame_data is a dictionary mapping frame_idx -> (num_objects, masks, bounding_boxes)

    # Initialize the captioner based on the configuration
    captioner_name = args.captioner
    device = config["device"]
    # Pass tot_vram_gb from args to select_captioner
    captioner = select_captioner(captioner_name, tot_vram_gb=args.tot_vram_gb, device=device)

    # Determine if masks should be used (from config)
    use_masks = args.use_masks

    # Create output directory with subfolder based on mask usage
    mask_subfolder = "with_masks" if use_masks else "without_masks"
    cap_out_dir = os.path.join(f"output/{dataset}/caption", captioner_name, mask_subfolder)
    if not os.path.exists(cap_out_dir):
        os.makedirs(cap_out_dir)

    # Initialize DataFrame to store all captions
    # [Modified] When initializing the DataFrame, the column names were expanded by avg_logprob, perplexity, and token_logprobs to store new statistical information
    all_captions_df = pd.DataFrame(columns=["frame_idx", "object_idx", "caption", "avg_logprob", "perplexity", "token_logprobs"])

    data_dir = f"data/{dataset}"

    # Process all frames
    print(f"Generating captions for {len(frame_data)} frames with mask usage: {use_masks}...")
    for frame_idx in tqdm(sorted(frame_data.keys())):
        num_objects, masks, bounding_boxes = frame_data[frame_idx]

        # Format frame_idx for loading the image
        formatted_frame_idx = "{:04d}".format(frame_idx)

        # Load the image
        image_path = os.path.join(data_dir, f"frame{formatted_frame_idx}.png")
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found. Skipping frame {formatted_frame_idx}.")
            continue

        image = Image.open(image_path).convert("RGB")

        # Generate captions for the objects in this frame
        if bounding_boxes:  # Only process if there are objects
            # Only pass masks if use_masks is True
            masks_to_use = masks if use_masks else None
            # [Modified] Here, two variables are needed to receive the text list and the statistics list returned by generate_captions
            captions, caption_stats = generate_captions(captioner_name, captioner, image, bounding_boxes, masks_to_use, user_prompt=user_prompt_vlm)

            # Add captions to DataFrame
            # [Modified] The original enumerate(captions) loop has been expanded to extract statistical variables
            for i, caption_text in enumerate(captions):
                # [Modified] Retrieve the statistical data of the current corresponding object
                stats = caption_stats[i]

                df_data = {
                    "frame_idx": formatted_frame_idx,
                    "object_idx": i,
                    "caption": caption_text,
                    # [Modified] Store the parsed average logarithmic probability and confusion degree in the dictionary
                    "avg_logprob": stats["avg_logprob"],
                    "perplexity": stats["perplexity"],
                    # [Modified] Since token_logprobs is typically a list/dictionary structure and cannot be directly saved as text in CSV, Json.dumps is used to serialize it into a JSON string before saving
                    "token_logprobs": json.dumps(stats["token_logprobs"]) if stats["token_logprobs"] is not None else None
                }

                all_captions_df = pd.concat([all_captions_df, pd.DataFrame([df_data])], ignore_index=True)
        else:
            print(f"\nNo objects detected in frame {formatted_frame_idx}")

    # Save all captions to CSV
    output_file = os.path.join(cap_out_dir, "all_captions.csv")
    all_captions_df.to_csv(output_file, sep=";", index=False)
    captioner.stop()  # Stop the captioner and release resources

    print(f"All captions saved to {output_file}")


if __name__ == "__main__":
    caption()