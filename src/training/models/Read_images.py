import os
import numpy as np
#import pydicom
from PIL import Image


from skimage.transform import resize

def create_array(path):
    #img = pydicom.dcmread(path)
    img = Image.open(path)
    img_arr = np.asarray(img).reshape(28,28,1)
    return img_arr

def read_img_to_array(root_dir,size):
    images_rcc = []
    images_lcc = []
    images_rmlo = []
    images_lmlo = []
    labels_rcc = []
    labels_lcc = []
    labels_rmlo = []
    labels_lmlo = []

    patients = [directory for directory in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, directory))]
    for id, patient in enumerate(patients):
        rcc = []
        lcc = []
        rmlo = []
        lmlo  = []
        dir_path = os.path.join(root_dir, patient)
        sessions = [directory for directory in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, directory))]
        for session in sessions:
            ses_path = os.path.join(dir_path, session)
            images = [file for file in os.listdir(ses_path) if os.path.isfile(os.path.join(ses_path, file)) and not file.startswith('.')]
            for image in images:
                try:
                    img_in = create_array(os.path.join(ses_path, image))/255
                    img_in = resize(img_in, (size, size))
                    #img_in = np.expand_dims(img_in, axis=2)                  
                    if 'RCC' in image.upper():
                        rcc.append(img_in)
                    elif 'LCC' in image.upper():
                        lcc.append(img_in)
                    elif 'RMLO' in image.upper():
                        rmlo.append(img_in)
                    elif 'LMLO' in image.upper():
                        lmlo.append(img_in)
                except:
                    print(os.path.join(ses_path, image))
                    
        for i in range(len(rcc)):
            for j in range(i+1,len(rcc)):
                temp = np.zeros((2, size, size,1))
                temp[0] = rcc[i]
                temp[1] = rcc[j]
                labels_rcc.append(id)
                images_rcc.append(temp)
        for i in range(len(lcc)):
            for j in range(i+1,len(lcc)):
                temp = np.zeros((2, size, size,1))
                temp[0] = lcc[i]
                temp[1] = lcc[j]
                labels_lcc.append(id)
                images_lcc.append(temp)
        for i in range(len(rmlo)):
            for j in range(i+1,len(rmlo)):
                temp = np.zeros((2, size, size,1))
                temp[0] = rmlo[i]
                temp[1] = rmlo[j]
                labels_rmlo.append(id)
                images_rmlo.append(temp)
        for i in range(len(lmlo)):
            for j in range(i+1,len(lmlo)):
                temp = np.zeros((2, size, size,1))
                temp[0] = lmlo[i]
                temp[1] = lmlo[j]
                labels_lmlo.append(id)
                images_lmlo.append(temp)

    rcc_array = np.zeros((len(images_rcc), 2, size, size, 1))
    for i, img in enumerate(images_rcc):
        rcc_array[i] = img  
    
    lcc_array = np.zeros((len(images_lcc), 2, size, size, 1))
    for i, img in enumerate(images_lcc):
        lcc_array[i] = img 
        
    rmlo_array = np.zeros((len(images_rmlo), 2, size, size, 1))
    for i, img in enumerate(images_rmlo):
        rmlo_array[i] = img 
        
    lmlo_array = np.zeros((len(images_lmlo), 2, size, size, 1))
    for i, img in enumerate(images_lmlo):
        lmlo_array[i] = img 
    
    return rcc_array, lcc_array, rmlo_array, lmlo_array, labels_rcc, labels_lcc, labels_rmlo, labels_lmlo