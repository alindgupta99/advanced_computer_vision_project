import os
import shutil
import constants
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import json

def collate_sketches_in_one_folder(src_sketch_folder_path, dest_sketch_folder_path):
    """
    Args:
    src_sketch_folder_path (String): Path to folder containing all folders of sketches 
    dest_sketch_folder_path (String): Path to folder where all the sketches will be placed
    """

    # Walk through the subfolders inside f1 (excluding f1_last itself)
    for cat_subfolder in os.listdir(src_sketch_folder_path):
        cat_subfolder_path = src_sketch_folder_path + '/' + cat_subfolder
            
        # Check if it's a directory and not all_photos
        if os.path.isdir(cat_subfolder_path) and cat_subfolder != constants.all_photos_folder_sub_path:
            for file in os.listdir(cat_subfolder_path):
                if file.endswith(constants.png_file_type):  # Check for .png files
                    file_path = os.path.join(cat_subfolder_path, file)
                    shutil.copy(file_path, dest_sketch_folder_path)  # Copy file to all_photos

def collate_sketches_in_one_folder_all_augmentations(src_sketch_folder_path):
    """
    Args:
    src_sketch_folder_path (String): Path to src folder containing images of all augmentations
    """

    for augmentation_folder in os.listdir(src_sketch_folder_path):
        dest_sketch_folder_path = src_sketch_folder_path + '/' + augmentation_folder + '/' + constants.all_photos_folder_sub_path        
        collate_sketches_in_one_folder(src_sketch_folder_path + '/' + augmentation_folder, dest_sketch_folder_path)

def read_image(image_path):
    """
    Args:
    image_path (String): Path to input image

    Returns:
    (Image object): Image object processed using PIL.Image class
    """

    image = Image.open(image_path).convert(constants.image_colour_code)
    return image

def preprocess_sketch_metadata(sketch_metadata_df):
    """
    Args:
    sketch_metadata_df (pd.Dataframe): dataframe object on which the preprocessing steps should be applied.
    """

    sketch_metadata_df[constants.sketch_filename_column_id] = sketch_metadata_df[constants.imagenet_id_column_id].astype(str) + constants.hiphen_connector + sketch_metadata_df[constants.sketch_id_column_id].astype(str)
    sketch_metadata_df[constants.worker_tag_column_id] = sketch_metadata_df[constants.worker_tag_column_id].fillna(sketch_metadata_df[constants.category_column_id])

def train_test_split_data(sketch_metadata_df):
    """
    Args:
    sketch_metadata_df (pd.Dataframe): dataframe object on which the train_test_split should be applied.

    Returns:
    (pd.Dataframe, pd.Dataframe): train and test split of the input data
    """

    train_sketch_metadata_df, test_sketch_metadata_df = train_test_split(sketch_metadata_df,
                                                                         test_size=constants.test_data_frac)
    
    return train_sketch_metadata_df, test_sketch_metadata_df

def dataframe_to_hf_dataset(df):
    """
    Args:
    df (df.DataFrame): dataframe object which should be converted to Hugging Face Dataset format.
    """

    return Dataset.from_pandas(df)

def drop_error_or_ambiguous_sketches(sketch_metadata_df):
    """
    Args:
    sketch_metadata_df (df.DataFrame): dataframe object where the error rows should be dropped.
    """
    filtered_df = sketch_metadata_df[(sketch_metadata_df[constants.has_error_column_id] != 1) & (sketch_metadata_df[constants.has_ambiguity_column_id] != 1)]
    return filtered_df

def read_sketch_metadata(metadata_file_path, preprocess_data=True):
    """
    Args:
    metadata_file_path (String): Path to sketch metadata file
    preprocess_data (boolean): Flag to indicate if the metadata df should be preprocessed before returning. 
                               Defaulted to True

    Returns:
    (pd.DataFrame): dataframe object with the sketch metadata
    """
    
    sketch_metadata_df = pd.read_csv(metadata_file_path)

    if (preprocess_data):
        preprocess_sketch_metadata(sketch_metadata_df)
    
    return sketch_metadata_df

def save_output(model_id, df_output, filtered_df_output):
    """
    Args:
    model_id (String): Indicates output belongs to which model
    df_output (dict): dictionary object containing generated output on full data
    filtered_df_output (dict): dictionary object containing generated output on filtered data

    Returns
    (String, String): Tuple of saved paths of full data and filtered data output
    """

    df_output_save_path = constants.save_dir_path + '/' + model_id + constants.full_data_model_prefix + constants.txt_file_type
    filtered_df_output_save_path = constants.save_dir_path + '/' + model_id + constants.filtered_data_model_prefix + constants.txt_file_type
    
    with open(df_output_save_path, constants.write_mode) as f:
        json.dump(df_output, f, indent=4)

    with open(filtered_df_output_save_path, constants.write_mode) as f:
        json.dump(filtered_df_output, f, indent=4)

    return df_output_save_path, filtered_df_output_save_path