#ifndef TRAINSYSTEM_H
#define TRAINSYSTEM_H
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class trainSystem
{
    public:
        trainSystem();
        vector<Mat> train(vector<vector<float> > regionFeatures);
    protected:
    private:
        double calcNorm(Mat inp);
        double accuracy(int length);
        Mat findSCombination(Mat currentFeatures,int numCenters,vector<int> &gamma);
};

#endif // TRAINSYSTEM_H
