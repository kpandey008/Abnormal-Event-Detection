#include "featureExtractor.h"
#include "VideoTrainer.h"
#include "trainSystem.h"
#include <vector>
#include <math.h>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

string videoDir;
VideoTrainer::VideoTrainer(string dir)
{
    videoDir=dir;
}
vector<string> VideoTrainer::loadFromDir(){

    vector<string> fileNames;
    string temp;
    DIR *dir;
    struct dirent *ent;
    dir = opendir (videoDir.c_str());
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
void VideoTrainer::trainAllVideos(){

    ///load the videos from the dir
    vector<string> video_files=loadFromDir();

    ///extract the features from all the videos
    cout << "--------------Feature Extraction Started-------------" << endl;
    VideoCapture video;
    vector<vector<int> > Loc;
    vector<vector<float> > features;
    featureExtractor extractor;
    int length=500;
    int numEachVol=7000;
    int curFeaNum=0;
    vector<vector<float> > allFeatures;

    for(int i=0;i<video_files.size();i++){

        video.open(videoDir+"/"+video_files[i]);
        features=extractor.extract(video,Loc);

        //normalize
        Mat featureMat(length,features.size(),CV_32F);
        for(int j=0;j<featureMat.cols;j++){
            for(int k=0;k<featureMat.rows;k++){
                featureMat.at<float>(k,j)=features[j].at(k);
            }
        }
        normalize(featureMat,featureMat,0,1,NORM_L2,CV_32F);

        //generate the random permutation
        vector<int> indices;
        for(int j=0;j<featureMat.cols;j++)
            indices.push_back(j);
        //shuffle
        random_shuffle(indices.begin(),indices.end());

        //selecting the features
        curFeaNum=min(numEachVol,(int)features.size());

        for(int j=0;j<curFeaNum;j++){
            allFeatures.push_back(features[indices[j]]);
        }
        cout << "Feature Extraction done for " << i <<"th video" << endl;
    }
    cout << "------------Feature Extraction Completed--------------" << endl;
    //principal component analysis
    Mat allfeatureMat(length,allFeatures.size(),CV_32F);
        for(int i=0;i<allfeatureMat.cols;i++){
            for(int j=0;j<allfeatureMat.rows;j++){
                allfeatureMat.at<float>(j,i)=allFeatures[i].at(j);
            }
        }
    int PCADim=100;
    PCA pca(allfeatureMat,Mat(),CV_PCA_DATA_AS_COL,PCADim);
    Mat projectionResult=pca.project(allfeatureMat);

    cout << projectionResult.size() << endl;
    cout << pca.eigenvectors.size() << endl;
    //storing the pca components to a file
    cout << "File writing started"<< endl;
    FileStorage storage("PCAComponents.yml",FileStorage::WRITE);
    storage << "PCA" << pca.eigenvectors;
    cout << "PCA File writing successful"<< endl;
    storage.release();

    ///find the sparse combinations for the training videos
    cout << "----------------Training phase started-------------------" << endl;
    vector<vector<float> > featureCollection;
    vector<float> temp;
    for(int i=0;i<projectionResult.cols;i++){
        for(int j=0;j<projectionResult.rows;j++){
            temp.push_back(projectionResult.at<float>(j,i));
        }
        featureCollection.push_back(temp);
        temp.clear();
    }

    trainSystem trainer;
    vector<Mat> sparseCombinations=trainer.train(featureCollection);
    cout << "----------------Combination Sets found-------------------" << endl;

    ///find the auxillary matrices from the sparse combinations
    Mat identityMat=Mat::eye(100,100,CV_32FC1);
    vector<Mat> auxillaryMatrices;
    Mat tmp_matrix,tmp_comb,tmp_res;
    Mat result;
    for(int i=0;i<sparseCombinations.size();i++){
        tmp_comb=sparseCombinations[i];
        tmp_matrix=tmp_comb * ((tmp_comb.t() * tmp_comb).inv());
        tmp_res=tmp_matrix * tmp_comb.t() - identityMat;
        result=tmp_res.clone();
        auxillaryMatrices.push_back(result);
    }

    ///File writing routines
    //write the obtained sparse combinations to the file
    FileStorage sparseStorage("SparseSet.yml",FileStorage::WRITE);
    sparseStorage << "Combinations" << sparseCombinations;
    sparseStorage.release();

    //write the auxillary matrix to the file
    FileStorage aux("Auxillary.yml",FileStorage::WRITE);
    aux << "Auxillary" << auxillaryMatrices;
    aux.release();

    cout << "File writing procedures completed" << endl;
}
