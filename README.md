# Abnormal-Event-Detection_IKST

This project is an OpenCV based implementation of the paper 'Abnormal Event Detection at 150FPS in Matlab' by Cewu Lu. Jianping Shi. Jiaya Jia published in ICCV'13. The algorithm has been implemented in C++ using the OpenCV library setup in CodeBlocks developement environment.
The link to the paper is http://ieeexplore.ieee.org/document/6751449/

### Prerequisites

For running the project, OpenCV needs to be setup. There are many blogs and resources available for setting up OpenCV either in Windows or Linux OS. I prefer Linux as everything is much easier to set up :P.

### How to use the code
The videos used for training and testing the sparse coding model have not been incorporated in the repository. However the Avenue dataset has been used to train the model for detecting abnormal events and is available from the following links
http://appsrv.cse.cuhk.edu.hk/~cwlu/Anormality_1000_FPS/dataset.html or
http://www.cse.yorku.ca/vision/research/anomalous-behaviour-data

Assuming that the prerequisites are in place, please follow the steps to run the project

1) Open the file main.cpp
2) If you want to retrain the model, uncommment the first half of the code in main.cpp and change the directory name depending on the location of the dataset in your computer. Otherwise follow step 3 for only testing the model

3) If you want to test the model with the existing parameters, place the videos you want to test with in the testing_videos folder in the project and run the program in CodeBlocks. 
