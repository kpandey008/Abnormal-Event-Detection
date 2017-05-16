#ifndef TESTSYSTEM_H
#define TESTSYSTEM_H
#include <vector>
#include <string>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


class testSystem
{
    public:
        testSystem();
        void test(string dir);
    protected:
    private:
        double calcNorm(Mat inp);
        vector<bool> calcError(Mat featuresPCA,vector<Mat> auxillary,double thresh_error);
        vector<string> loadFromDir(string directory);
};

#endif // TESTSYSTEM_H
