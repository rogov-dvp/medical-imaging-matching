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
# import query functions to database as well.

# FUNCTIONS:
#TODO: Query for database for preprocessed version of the given unpreprocessed path
def query(up_file):
  pp_file = ""
  print("querying" + up_file)
  return pp_file

#TODO: Send unprocessed path (up_file), preprocess it, and return the pp_file of that image
def preprocess(up_file):
  pp_file = ""
  print("querying" + up_file)
  return pp_file

#TODO: add preprocessed path to database.
def add_db(pp_file):
  print(pp_file + " added to database")


# CODE
# Get array of string from unprocessed_paths.txt
pp_paths = []
up_paths = []   #unprocessed paths
with open('unprocessed_paths.txt') as f:
    up_paths = f.readlines()

# Check for 
# 1. existing preprocessed files from "database" 
# OR
# 2. send to preprocessing component
for up_file in up_paths: 
  pp_file = query(up_file)
  if not pp_file:
    # pp_file string is empty. Send to preprocessing component
    pp_file = preprocess()
    # add these newly preprocessed files to database
    add_db(pp_file)

  #add file to array
  pp_paths.append(pp_file)



# Once all unprocessed paths (up_files) have been processed, send to matching algorithm:
#Temp, matching similarity function should probably be on a different file
def matching_sim():
  return 69.69  #nice

# Run Matching similarity algorithm function. We could insert some status_bar.py potentially?
percentage = matching_sim()

#Output
print("Similarity matching percentage: " + percentage + "%")

