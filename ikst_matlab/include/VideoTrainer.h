#ifndef VIDEOTRAINER_H
#define VIDEOTRAINER_H
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class VideoTrainer
{
    public:
        VideoTrainer(string dir);
        void trainAllVideos();
    protected:
    private:
    vector<string> loadFromDir();
    string videoDir;
};

#endif // VIDEOTRAINER_H
