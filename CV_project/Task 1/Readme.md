This file contains instructions pertinent to the code related to task 1

In the Task 1 directory we provide all the code and .ipynb files to run the code.
run.ipynb file runs all the modular code provided in this directory and generates the results
presented in the report and the presentation

demo.ipynb file is an independent file used for only demonstration purposes of the Task 1

In this directory, do not provide the data which is used for getting the results.
We provide the following steps to extract the data and place it within the Task 1 directory:-

1. Download the data from this url: https://sketchy.eye.gatech.edu/ and place it in a folder named Data
parallel to the task 1 directory (or any other directory but the relevant path will need to be changed in the constants.py file)

2. Use the collate_sketches_in_one_folder_all_augmentations() method in the data_utils.py file to extract all sketches of all augmentations.

3. Rename the folder with the augmentation named "tx_000000000000" to all_photos to be able to reproduce the results faithfully.

Because of resource constraints, this code was run on google colab and therefore we highly advise to
run the code on google colab to be able to run it smoothly. Otherwise the src directory paths will be required
to change in the constants.py file.