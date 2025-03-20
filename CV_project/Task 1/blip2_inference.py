import constants
import data_utils
import torch

def generate_blip2_caption(model, processor, image, device, prompt):
    """
    Args:
    model (transformers.Blip2ForConditionalGeneration): Blip2 model used for processing the input image
    processor (transformers.AutoProcessor): Blip2 model processor used to decode model output
    image (PIL.Image): input image
    device (String): device on which to host the model
    prompt (String): Input prompt for caption generation

    Returns:
    (String): Generated image caption
    """
    inputs = processor(image, text=prompt, return_tensors=constants.blip2_return_tensor).to(device, torch.float16)    

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=constants.blip2_max_new_tokens)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def generate_blip2_caption_all(model, processor, samples, device, should_prompt=False):
    """
    Args:
    model (transformers.Blip2ForConditionalGeneration): Blip2 model used for processing the input image
    processor (transformers.AutoProcessor): Blip2 model processor used to decode model output
    samples (datasets.Dataset): dataset object containing image paths.
    device (String): device on which to host the model
    should_prompt (boolean): indicates if an additional input prompt should be added or not

    Returns:
    (List(String), List(String), List(String)): List of reference captions, 
                                                List of image paths and List of generated captions
    """

    reference_captions = []
    image_paths = []
    generated_captions = []

    for sample in samples:
        image_path = constants.sketch_all_photos_path + '/' + sample[constants.sketch_filename_column_id] + constants.png_file_type
        image = data_utils.read_image(image_path)
        if (should_prompt):
            generated_caption = generate_blip2_caption(model, processor, image, device, constants.blip2_input_prompt)
        else:
            generated_caption = generate_blip2_caption(model, processor, image, device, None)
        reference_captions.append(sample[constants.worker_tag_column_id])
        image_paths.append(image_path)
        generated_captions.append(generated_caption)

    return (reference_captions, image_paths, generated_captions)