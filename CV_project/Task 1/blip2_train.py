import constants
import data_utils
import blip2_inference
import evaluate
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
import pickle
import time

def initialize_blip2_model():
    """
    Returns:
    (transformers.AutoProcessor, transformers.Blip2ForConditionalGeneration): The processor and the model respectively.
    """

    blip2_processor = AutoProcessor.from_pretrained(constants.blip2_model_name)
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(constants.blip2_model_name,
                                                                torch_dtype=torch.float16,
                                                                device_map=constants.blip2_device_map)
    return blip2_processor, blip2_model

# DataLoader function
def collate_fn(samples, processor, device):
    """
    Args:
    samples (datasets.Dataset): dataset object.
    processor (transformers.AutoProcessor): blip2 processor object
    device (String): Indicates which hardware to use

    Returns:

    """
    images = []
    texts = []
    for sample in samples:
        image_path = constants.sketch_all_photos_path + '/' + sample[constants.sketch_filename_column_id] + constants.png_file_type
        image = data_utils.read_image(image_path)
        images.append(image)
        texts.append(constants.image_tag + sample[constants.worker_tag_column_id])
    return processor(images=images, text=texts, return_tensors=constants.llava_return_tensor, padding=True).to(device)

def run(device, should_prompt):
    """
    Args:
    device (String): Indicates where hardware to use
    should_prompt (boolean): indicates if an additional input prompt should be added or not

    Returns
    (String, String): Tuple of saved paths of blip2 processor and blip2 model
    """

    blip2_processor, blip2_model = initialize_blip2_model()

    # Read metadata file
    df = data_utils.read_sketch_metadata(constants.sketch_metadata_file_path)

    # Generate dataset with filtered data
    filtered_df = data_utils.drop_error_or_ambiguous_sketches(df)

    # Reducing dataset for computational efficiency
    df = df.sample(n=constants.blip2_num_reduced_rows)
    filtered_df = filtered_df.sample(n=constants.blip2_num_reduced_rows)

    df_dataset = data_utils.dataframe_to_hf_dataset(df)
    filtered_df_dataset = data_utils.dataframe_to_hf_dataset(filtered_df)

    # Generate df and filtered performances
    df_bert_score, df_image_descriptiveness_score, df_output = evaluate_model(blip2_model, blip2_processor,
                                                                              df_dataset, device, should_prompt)
    print('Unfiltered data Performance:')
    print('Bert Score: ' + str(df_bert_score))
    print('Image descriptiveness score: ' + str(df_image_descriptiveness_score))
    filtered_df_bert_score, filtered_df_image_descriptiveness_score, filtered_df_output = evaluate_model(
        blip2_model, blip2_processor, filtered_df_dataset, device, should_prompt)
    print('Filtered data Performance::')
    print('Bert Score: ' + str(filtered_df_bert_score))
    print('Image descriptiveness score: ' + str(filtered_df_image_descriptiveness_score))

    # Save generated output 
    if (should_prompt):
        return data_utils.save_output(constants.blip2_prompted_model_id, df_output, filtered_df_output)
    else:
        return data_utils.save_output(constants.blip2_unprompted_model_id, df_output, filtered_df_output)

def evaluate_model(model, processor, samples, device, should_prompt):
    """
    Args:
    model (transformers.LlavaForConditionalGeneration): Llava model used for processing the input image
    processor (transformers.AutoProcessor): Llava model processor used to decode model output
    samples (datasets.Dataset): dataset object containing image paths.
    device (String): device on which to host the model
    should_prompt (boolean): indicates if an additional input prompt should be added or not

    Returns:
    (float, float, dict): Bert score, image descriptiveness score and generated output dictionary
    """

    reference_captions, image_paths, generated_captions = blip2_inference.generate_blip2_caption_all(
        model, processor, samples, device, should_prompt)
    outputs = {}
    outputs[constants.reference_captions_key] = reference_captions
    outputs[constants.image_paths_key] = image_paths
    outputs[constants.generated_captions_key] = generated_captions
    bert_score = evaluate.bertscore(reference_captions, generated_captions)
    image_descriptiveness_score = evaluate.image_descriptiveness_score(image_paths, generated_captions)
    return (bert_score, image_descriptiveness_score, outputs)