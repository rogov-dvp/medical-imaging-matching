**Data Flow Diagrams**

***DFD Level 0***

![](https://raw.githubusercontent.com/rogov-dvp/medical-imaging-matching/main/docs/project_requirements/images/dfd_level0.jpg)


The client will query and receive the specified mammograms which are stored in the File System. The client will then send at least two mammograms to the Image Matching Script. This Image Matching script will then return a match similarity percentage to the client. 


***DFD Level 1***

![](https://raw.githubusercontent.com/rogov-dvp/medical-imaging-matching/main/docs/project_requirements/images/dfd_level1.jpg)


The client will query and receive the specified mammograms which are stored in the File System.

The client sends at least two mammograms to the Image Matching Script component. It will then query the Database/Files to see if the preprocessed version of the mammogram exists. If it does, it will directly send them to the Matching Similarity Algorithm component. If not, This script will then send one mammogram to the Preprocessing Algorithm component. 

The Preprocessing Algorithm component then returns a preprocessed mammogram to the Image Matching Script. The Image Matching Script will then save it to the Database/File System. Once the Image Matching script has two preprocessed mammograms, it will send them to the Matching Similarity Algorithm component to calculate a matching similarity percentage. An overall percentage will be returned to the Image Matching Script component. Finally, the Image Matching Script will then return this match similarity percentage to the client.
