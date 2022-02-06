import pandas as pd
import numpy as np
import os
from PIL import Image
from pathlib import Path
import xml.etree.cElementTree as et

DIRECTORY = 'img_data'

#make folder for cropped images
# Path("/img_data_cropped").mkdir(parents=True, exist_ok=True)

#iterate through img/xml folder where DIRECTORY is the directory
for filename in os.listdir(DIRECTORY):
    f = os.path.join(DIRECTORY, filename)
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    if os.path.isfile(f) and f.endswith(".xml"):
        #each .xml file, get coordinates
        tree = et.parse(f)
        root = tree.getroot()
        for xmin in root.iter('xmin'):
            x_min = int(xmin.text,0)
        for ymin in root.iter('ymin'):
            y_min = int(ymin.text,0)
        for xmax in root.iter('xmax'):
            x_max = int(xmax.text,0)
        for ymax in root.iter('ymax'):
            y_max = int(ymax.text,0)
        
        # split file name by slash
        parsedFileName = f.split('/')
        
        # index for last parse
        n = len(parsedFileName)-1 

        #get image file name with .jpg extension
        imageFile = parsedFileName[n].replace('.xml','.jpg')

        #crop the image if the .xml file matches the same name as the .jpg file
        if os.path.exists(DIRECTORY + "/" + imageFile):
            #crop image and save in specified directory.
            img = Image.open(parsedFileName[0]+ "/" + imageFile)
            img2 = img.crop((x_min,y_min,x_max,y_max))
            img2.save("img_data_cropped/" + imageFile)        
    