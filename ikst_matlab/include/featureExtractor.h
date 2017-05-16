#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class featureExtractor
{
    public:
        featureExtractor();
        vector<vector<float> > extract(VideoCapture video,vector<vector<int> > &loc);
    protected:
    private:
        Mat conv2(Mat img,Mat kernel);
};

#endif // FEATUREEXTRACTOR_H
