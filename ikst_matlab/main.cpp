#include "VideoTrainer.h"
#include "testSystem.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    /*string str="D:/btp_iitbbs/Abnormal Event detection at 150 FPS in MATLAB/Datasets/Avenue Dataset/training_videos";
    VideoTrainer trainer(str);
    //vector<vector<float> > features;
    //features=extractor.extract(cap);
    trainer.trainAllVideos();
    //cout << features.size() << endl;*/

    //testing the video
    cout << "Testing phase started" << endl;
    testSystem tester;
    string dir="testing_videos";
    tester.test(dir);

    waitKey(0);
    return 0;
}



