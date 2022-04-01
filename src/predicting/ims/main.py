# input of file paths to unprocessed mammorgrams
# by alex rogov
#
# I am assuming that user's unprocessed path to images are placed in unprocessed_path.txt.
#
# A few options here for input:
# 1. Query for preprossing files
# 2. Accept files from preprocessing component
#

# TODO: get the functions for these two
# import preprocessing component function
import os
import sys
import csv

sys.path.append("../preprocessing/")
from preprocess_data import PreprocessData

sys.path.append("../src/models")
# TODO: import models

pp_filepath = "../test_kaggle_images/processed_images/"
processed_path = "../test_images_kaggle/processed_images"
unprocessed_images = "../test_images_kaggle/images"

# FUNCTIONS:
def query(up_file):
    """
  This function searches the processed_images files for a match, and returns any matches if there are any.

  Args:
      up_file (string): filename we are searching for.

  Returns:
      list: list of filenames that are matches, if there are no matches it is an empty list.
  """
    files = []
    print("querying " + str(up_file))
    for root, dirs, files in os.walk(pp_filepath):
        for file in files:
            if file.lower() == up_file.lower():
                files.append(file)
    return files


# TODO: preprocess.py needs to be fully implemented
def preprocess(up_file):
    """
  This function sends the unprocessed file for preprocessing.

  Args:
      up_file (string): file path of the unprocessed file.

  Returns:
      string: filepath of the now processed image file.
  """
    pp_file = ""
    print("querying " + str(up_file))
    ppd = PreprocessData(up_file, processed_path, unprocessed_images)
    result = ppd.process_image()
    print(result)
    print(type(result))
    if result.lower() != up_file.lower():
        pp_file = result
    return pp_file


# TODO: send filepaths to models, models need to be refactored into classes and to reflect these inputs
def matching_sim(files):
    """
  This function calls upon the models and returns a similarity score.

  Returns:
      [double]: [similarity score]
  """
    return 69.69  # nice


# CODE
# TODO port preprocessing, model, breast detection and batch builder into main.

# read in mismatch
mismatches = []

with open("../data/mismatches.csv", "r") as file:
    csv_file = csv.reader(file)

    for line in csv_file:
        mismatches.append(line)

# Now that we have our mismatches we can
# Check for
# 1. existing preprocessed files from "database"
# OR
# 2. send to preprocessing component
for mismatch in mismatches:
    files = []
    for file in mismatch:
        query_result = query(file)
        if len(query_result) == 0:
            # pp_file string is empty. Send to preprocessing component
            filepath = preprocess(file)
            if len(filepath) != 0:
                # add file to array
                files.append(file)
        else:
            files.append(query_result[0])

    # Run Matching similarity algorithm function. We could insert some status_bar.py potentially?
    percentage = matching_sim(files)

    # Output
    print("Similarity matching percentage: " + str(percentage) + "%")
