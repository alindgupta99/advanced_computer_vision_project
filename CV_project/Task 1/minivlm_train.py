import constants
import data_utils
import minivlm_inference
import evaluate
from transformers import AutoProcessor, AutoModel
import torch
from torch.utils.data import DataLoader
from PIL import Image

def initialize_minivlm_model():
    """
    Returns:
    (transformers.AutoProcessor, transformers.AutoModelForSeq2SeqLM): The processor and the model respectively.
    """
    minivlm_processor = AutoProcessor.from_pretrained(constants.minivlm_model_name, use_fast=True)
    minivlm_model = AutoModel.from_pretrained(
        constants.minivlm_model_name,
        torch_dtype=torch.float64,
        device_map=constants.minivlm_device_map
    )
    return minivlm_processor, minivlm_model

def save_model_config(minivlm_processor, minivlm_model, save_filename, save_dir_path):
    """
    Args:
    minivlm_processor (transformers.AutoProcessor): The MiniVLM processor to be saved
    minivlm_model (transformers.AutoModelForSeq2SeqLM): The MiniVLM model to be saved
    save_filename (String): Unique filename of the saved files
    save_dir_path (String): The path to the directory where the files need to be saved

    Returns:
    (String, String): Tuple of saved paths of MiniVLM processor and MiniVLM model
    """
    minivlm_processor_save_path = f"{save_dir_path}/{save_filename}{constants.minivlm_processor_suffix}"
    minivlm_model_save_path = f"{save_dir_path}/{save_filename}{constants.minivlm_model_suffix}"
    
    minivlm_processor.save_pretrained(minivlm_processor_save_path)
    minivlm_model.save_pretrained(minivlm_model_save_path)

    return minivlm_processor_save_path, minivlm_model_save_path

# DataLoader function
def collate_fn(samples, processor, device):
    """
    Args:
    samples (datasets.Dataset): dataset object.
    processor (transformers.AutoProcessor): MiniVLM processor object
    device (String): Indicates which hardware to use

    Returns:
    Dictionary with tokenized images and texts ready for model input
    """
    images = []
    texts = []
    for sample in samples:
        image_path = f"{constants.sketch_all_photos_path}/{sample[constants.sketch_filename_column_id]}"
        image = data_utils.read_image(image_path)
        images.append(image)
        texts.append(sample[constants.worker_tag_column_id])
    
    return processor(images=images, text=texts, return_tensors=constants.minivlm_return_tensor).to(device)

def train_and_save(finetune, device, filter_data=False):
    """
    Args:
    finetune (boolean): Flag to indicate if the model should be finetuned with our data
    filter_data (boolean): Flag to indicate if the data should be filtered before training
    device (String): Indicates which hardware to use

    Returns:
    (String, String): Tuple of saved paths of MiniVLM processor and MiniVLM model
    """
    minivlm_processor, minivlm_model = initialize_minivlm_model()
    model_save_filename = None

    # Read metadata file
    df = data_utils.read_sketch_metadata(constants.sketch_metadata_file_path)

    # Filter data if needed
    if filter_data:
        model_save_filename = constants.filtered_data_model_prefix
        df = data_utils.drop_error_or_ambiguous_sketches(df)

    # Train-test split
    train_df, test_df = data_utils.train_test_split_data(df)
    train_dataset = data_utils.dataframe_to_hf_dataset(train_df)
    test_dataset = data_utils.dataframe_to_hf_dataset(test_df)

    if finetune:
        if model_save_filename is None:
            model_save_filename = constants.full_data_model_prefix

        dataloader = DataLoader(
            train_dataset,
            batch_size=constants.train_batch_size,
            shuffle=constants.should_shuffle,
            collate_fn=lambda x: collate_fn(x, minivlm_processor, device)
        )
        optimizer = torch.optim.AdamW(minivlm_model.parameters(), lr=constants.train_learning_rate)

        for epoch in range(constants.train_num_epochs):
            print(f"Epoch {epoch+1}/{constants.train_num_epochs} for {model_save_filename}")
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = minivlm_model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"{model_save_filename} Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")

    else:
        model_save_filename = constants.untrained_model_prefix

    # Generate train and test performances
    train_bert_score, train_image_descriptiveness_score = evaluate_model(
        minivlm_model, minivlm_processor, train_dataset, device
    )
    print('Training Performance:')
    print(f'Bert Score: {train_bert_score}')
    print(f'Image descriptiveness score: {train_image_descriptiveness_score}')

    test_bert_score, test_image_descriptiveness_score = evaluate_model(
        minivlm_model, minivlm_processor, test_dataset, device
    )
    print('Test Performance:')
    print(f'Bert Score: {test_bert_score}')
    print(f'Image descriptiveness score: {test_image_descriptiveness_score}')

    # Save the model after training
    return save_model_config(minivlm_processor, minivlm_model, model_save_filename, constants.model_save_dir_path)

def evaluate_model(model, processor, samples, device):
    """
    Args:
    model (transformers.AutoModelForSeq2SeqLM): MiniVLM model used for processing the input image
    processor (transformers.AutoProcessor): MiniVLM processor used to decode model output
    samples (datasets.Dataset): dataset object containing image paths.
    device (String): device on which to host the model

    Returns:
    (float, float): Bert score and image descriptiveness score
    """
    reference_captions, image_paths, generated_captions = minivlm_inference.generate_minivlm_caption_all(
        model, processor, samples, device
    )
    bert_score = evaluate.bertscore(reference_captions, generated_captions)
    image_descriptiveness_score = evaluate.image_descriptiveness_score(image_paths, generated_captions)
    return (bert_score, image_descriptiveness_score)
