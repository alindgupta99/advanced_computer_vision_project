import constants
import data_utils
import llava_inference
import evaluate
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

def initialize_llava_model():
    """
    Returns:
    (transformers.AutoProcessor, transformers.LlavaForConditionalGeneration): The processor and the model respectively.
    """

    llava_processor = AutoProcessor.from_pretrained(constants.llava_model_name, use_fast=True)
    llava_model = LlavaForConditionalGeneration.from_pretrained(constants.llava_model_name,
                                                                torch_dtype=torch.float64,
                                                                device_map=constants.llava_device_map)
    return llava_processor, llava_model

# DataLoader function
def collate_fn(samples, processor, device):
    """
    Args:
    samples (datasets.Dataset): dataset object.
    processor (transformers.AutoProcessor): llava processor object
    device (String): Indicates which hardware to use

    Returns:

    """
    images = []
    texts = []
    for sample in samples:
        image_path = constants.sketch_all_photos_path + '/' + sample[constants.sketch_filename_column_id]
        image = processor.image_processor(image_path)
        images.append(image)
        texts.append(sample[constants.worker_tag_column_id])
    return processor(images=images, text=texts, return_tensors=constants.llava_return_tensor).to(device)

def run(device):
    """
    Args:
    device (String): Indicates which hardware to use

    Returns
    (String, String): Tuple of saved paths of llava processor and llava_model
    """

    llava_processor, llava_model = initialize_llava_model()

    # Read metadata file
    df = data_utils.read_sketch_metadata(constants.sketch_metadata_file_path)
    
    # Generate dataset with filtered data
    filtered_df = data_utils.drop_error_or_ambiguous_sketches(df)
    
    # Reducing dataset for computational efficiency
    df = df.sample(n=constants.blip2_num_reduced_rows)
    filtered_df = filtered_df.sample(n=constants.blip2_num_reduced_rows)

    df_dataset = data_utils.dataframe_to_hf_dataset(df)
    filtered_df_dataset = data_utils.dataframe_to_hf_dataset(filtered_df)

    # Generate train and test performances
    df_bert_score, df_image_descriptiveness_score, df_outputs = evaluate_model(llava_model, llava_processor,
                                                                               df_dataset, device)
    print('Unfiltered data Performance:')
    print('Bert Score: ' + str(df_bert_score))
    print('Image descriptiveness score: ' + str(df_image_descriptiveness_score))
    filtered_df_bert_score, filtered_df_image_descriptiveness_score, filtered_df_outputs = evaluate_model(
        llava_model, llava_processor, filtered_df_dataset, device)
    print('Filtered data Performance:')
    print('Bert Score: ' + str(filtered_df_bert_score))
    print('Image descriptiveness score: ' + str(filtered_df_image_descriptiveness_score))

    # Save the output
    return data_utils.save_output(constants.llava_model_id, df_outputs, filtered_df_outputs)

def evaluate_model(model, processor, samples, device):
    """
    Args:
    model (transformers.LlavaForConditionalGeneration): Llava model used for processing the input image
    processor (transformers.AutoProcessor): Llava model processor used to decode model output
    samples (datasets.Dataset): dataset object containing image paths.
    device (String): device on which to host the model

    Returns:
    (float, float, dict): Bert score, image descriptiveness score and generated output dictionary 
    """

    reference_captions, image_paths, generated_captions = llava_inference.generate_llava_caption_all(
        model, processor, samples, device)
    outputs = {}
    outputs[constants.reference_captions_key] = reference_captions
    outputs[constants.image_paths_key] = image_paths
    outputs[constants.generated_captions_key] = generated_captions
    bert_score = evaluate.bertscore(reference_captions, generated_captions)
    image_descriptiveness_score = evaluate.image_descriptiveness_score(image_paths, generated_captions)
    return (bert_score, image_descriptiveness_score, outputs)
        
        
