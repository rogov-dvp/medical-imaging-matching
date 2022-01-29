# January Gitstats Report

1. Worked on getting a detection of 50 test mammograms.
  - January 28, 2021
  - Issue #1235
  - I have the code mostly set except I now need to find the dimensions of the detections and cut it out with pillow or opencv. My current issue is that my M1 macbook doesnt run tensorflow. I written down the initial steps I took when doing this just in case:   

https://www.youtube.com/watch?v=IOI0o3Cxv9Q

Specs used:

python 3.9.0
pip 21.x.x

Step for mask detection:
1. git clone https://github.com/nicknochnack/RealTimeObjectDetection.git
2. /RealTimeObjectDetection/Tensorflow
3. git clone https://github.com/tzutalin/labelImg.git
4. pip3 install PyQt5
5. pip3 install lmxl 
6. pyrcc5 -o resources.py resources.qrc
7. Move both files to labelImg/libs
8. python3 labelImg.py
9. View tab -> Auto Save Mode
10. Select where labelled images go to using Change Save Dir
11. Open images folder
12. Use “W” key to create box label. Use “A” and “D” to move between images. Add label desc to label.
12. Create labels on images
13. Add some to train folder and some to test folder
14. write the code..


to Train:
Takethe output of the code box under step 6. (“6. Train the Model”). Head to the root directory (RealTimeObjectDetection) and run it as a CLI. It should take a while. You’ll see a lot of output, occasionally, it will give a loss. Loss under 0.5 is good.

After, check how many checkpoints exists in Tensorflow/model/<model>/checkpoint