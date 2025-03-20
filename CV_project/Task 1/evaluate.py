import constants
from bert_score import score
from transformers import CLIPProcessor, CLIPModel
import torch
import data_utils

def bertscore(reference, generated):
    """
    Args:
    reference (List(String)): List of ground truth captions
    generated (List(String)): List of generated captions

    Returns:
    (float): F1 score of similarity scores between reference and generated captions
    """

    P, R, F1 = score(generated, reference, lang=constants.bert_score_lan, verbose=True)
    return F1.mean().item()

def initialize_clip_model(device):
    """
    Args:
    device (String): Indicates which hardware to use

    Returns:
    (clip.Model, clip.Processor): Returns the clip model and processor
    """

    clip_model = CLIPModel.from_pretrained(constants.clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(constants.clip_model_name)
    return (clip_model, clip_processor)

def clip_score(clip_model, clip_processor, image_path, caption):
    """
    Args:
    clip_model (clip.Model): The clip model used to process images
    clip_processor (clip.Processor): The clip processor used to process images
    image_path (String): Path to the input image
    caption (String): Generated caption for the image

    Returns:
    (float): Returns similarity score between image and generated caption
    """

    # Compute CLIP similarity score between the image and generated caption
    image = data_utils.read_image(image_path)
    
    # Preprocess the inputs (image and text)
    inputs = clip_processor(text=caption, images=image, return_tensors=constants.clip_return_tensors,
                            padding=True)

    # Forward pass through the model
    outputs = clip_model(**inputs)

    # Get the image and text embeddings
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # Normalize the embeddings
    image_embeds /= image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Calculate the cosine similarity between image and text embeddings
    similarity = (image_embeds @ text_embeds.T).squeeze()
    
    return similarity.item()

def image_descriptiveness_score(image_paths, captions):
    """
    Args:
    image_paths (List(String)): List of paths to images
    captions (List(String)): List of generated captions

    Returns:
    (float): Average similarity score between images and generated caption
    """

    device = constants.cuda_device if torch.cuda.is_available() else constants.cpu_device
    clip_model, clip_preprocess = initialize_clip_model(device)
    total_similarity = 0
    num_samples = len(image_paths)

    for idx in range(num_samples):
        total_similarity += clip_score(clip_model, clip_preprocess, image_paths[idx], captions[idx])

    avg_similarity = total_similarity / num_samples
    return avg_similarity