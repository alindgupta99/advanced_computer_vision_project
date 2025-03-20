import constants
import data_utils

def generate_minivlm_caption(model, processor, image, device):
    """
    Args:
    model (transformers.AutoModelForSeq2SeqLM): MiniVLM model used for processing the input image
    processor (transformers.AutoProcessor): MiniVLM model processor used to decode model output
    image (PIL.Image): input image
    device (String): device on which to host the model

    Returns:
    (String): Generated image caption
    """
    inputs = processor(images=image, text=constants.minivlm_input_prompt,
                       return_tensors=constants.minivlm_return_tensor).to(device)
    output = model.generate(**inputs)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption

def generate_minivlm_caption_all(model, processor, samples, device):
    """
    Args:
    model (transformers.AutoModelForSeq2SeqLM): MiniVLM model used for processing the input image
    processor (transformers.AutoProcessor): MiniVLM model processor used to decode model output
    samples (datasets.Dataset): dataset object containing image paths.
    device (String): device on which to host the model

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
        generated_caption = generate_minivlm_caption(model, processor, image, device)
        reference_captions.append(sample[constants.worker_tag_column_id])
        image_paths.append(image_path)
        generated_captions.append(generated_caption)

    return (reference_captions, image_paths, generated_captions)
