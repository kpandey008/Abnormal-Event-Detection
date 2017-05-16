#include "testSystem.h"
#include "testFeaExtractor.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>

using namespace std;
using namespace cv;

testSystem::testSystem()
{

}
double testSystem::calcNorm(Mat inp){
        double val=0;
        for(int i=0;i<inp.rows;i++){
            for(int j=0;j<inp.cols;j++){
                val+=(inp.at<float>(i,j)*inp.at<float>(i,j));
            }
        }
        return sqrt(val);
}
vector<string> testSystem::loadFromDir(string directory){

    vector<string> fileNames;
    string temp;
    DIR *dir;
    struct dirent *ent;
    dir = opendir (directory.c_str());
    if (dir!= NULL) {
        while (ent = readdir (dir)) {
            temp=ent->d_name;
            if(temp[0]!='.')
                fileNames.push_back(temp);
        }
        closedir(dir);
    }else {
        cout << "The directory could not be read" << endl;
        exit(0);
    }
    return fileNames;
}
vector<bool> testSystem::calcError(Mat featuresPCA,vector<Mat> auxillary,double thresh_error){

    vector<bool> result;
    int K=auxillary.size();
    Mat tmp_aux;
    Mat res;
    double norm,norm_sq;
    int flag=0;
    Mat tmp_feature;
    for(int i=0;i<featuresPCA.cols;i++){
        tmp_feature=featuresPCA.col(i);
        for(int j=0;j<K;j++){
            tmp_aux=auxillary[j].t();
            res=tmp_aux*tmp_feature;
            norm=calcNorm(res.clone());
            norm_sq=pow(norm,2);
            if(norm_sq < thresh_error){
                flag=1;
                break;
            }
        }
        if(flag==1)
            result.push_back(false);
        else
            result.push_back(true);
        flag=0;
    }
    return result;
}
void testSystem::test(string dir){

    double thresh_error=0.2; //error for result calculation
    ///find the files specified in the testing path
    vector<string> files=loadFromDir(dir);

    ///test the files in the directory
    testFeaExtractor extractor;
    Mat PCACoeff;
    FileStorage PCAReader("PCAComponents.yml",FileStorage::READ);
    FileStorage AuxillaryReader("Auxillary.yml",FileStorage::READ);

    cout << "File reading done" << endl;
    Mat PCA_comp;
    vector<Mat> auxillary;
    vector<vector<float> > features;
    vector<vector<int> > location;
    VideoCapture testVideo;
    for(int i=0;i<1;i++){//dir.size();i++){
        cout << "testing for video" << i << endl;
        string filePath=dir+"/"+files[i];
        testVideo.open(filePath);
        int numFrames=testVideo.get(CV_CAP_PROP_FRAME_COUNT);

        ///extract the features from the video
        features=extractor.extract(testVideo,location);
        Mat features_M(features[0].size(),features.size(),CV_32FC1);
        cout << "feature extraction done" << endl;

        for(int j=0;j<features.size();j++){
            for(int k=0;k<features[0].size();k++){
                features_M.at<float>(k,j)=features[j].at(k);
            }
        }
        ///compute the PCA for the extracted features
        //load the PCA components from a file
        PCAReader["PCA"] >> PCA_comp;
        cout << PCA_comp.size() << endl;
        cout << features_M.size() << endl;
        Mat featuresPCA=PCA_comp*features_M;
        cout << "PCA COMPLETED" << endl;

        ///load the auxillary matrices from the file
        AuxillaryReader["Auxillary"] >> auxillary;

        ///test the features for abnormal events
        vector<bool> result=calcError(featuresPCA,auxillary,thresh_error);
        int x,y,z;
        Mat AbEvent[numFrames];
        for(int ii=0;ii<numFrames;ii++)
            AbEvent[ii]=Mat::zeros(12,16,CV_32FC1);
        for(int ii=0;ii<result.size();ii++){
            x=location[ii].at(0);
            y=location[ii].at(1);
            z=location[ii].at(2);
            AbEvent[z].at<float>(x,y)=(float)result[ii];
        }
        cout << "Testing for video completed" << endl;
        //display the video to the user
        Mat result_frame,orig_frame;
        namedWindow("result");
        namedWindow("original");
        VideoCapture tester(filePath);
        for(int ii=0;ii<numFrames;ii++){

            resize(AbEvent[ii],result_frame,Size(160,120),INTER_CUBIC);
            tester >> orig_frame;
            imshow("result",result_frame);
            imshow("original",orig_frame);
            waitKey(40);
        }

    }
}
