#include "testFeaExtractor.h"
#include <vector>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

testFeaExtractor::testFeaExtractor()
{
    //ctor
}
Mat testFeaExtractor::conv2(Mat img, Mat kernel){

    ///compute the convolution for the matrix operation
    Mat dest;
    Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
    int borderMode = BORDER_CONSTANT;
    flip(kernel,kernel,-1);
    filter2D(img, dest, img.depth(), kernel, anchor, 0, borderMode);

    return dest;

}
vector<vector<float> > testFeaExtractor::extract(VideoCapture video,vector<vector<int> > &location){

    cout << "Feature extracion started" << endl;
    ///parameters
    int f_height=120;
    int BKH=12;
    int BKW=16;
    int f_width=160;
    int patchWin=10;
    int srs=5;     //spatial sampling rate
    int trs=2;     // temporal sampling rate
    int MT_thr=5;  // 3d patch selecting threshold
    int tprLen=5;  //temporal length

    ///determine the blurring kernel
    float blur_ar[3][3]={{0.0751,0.1238,0.0751},{0.1238,0.2042,0.1238},{0.0751,0.1238,0.0751}};
    Mat blurKer(3,3,CV_32F,blur_ar);
    Mat mask(120,160,CV_32FC1);
    //create the mask
    mask=conv2(Mat::ones(120,160,CV_32F),blurKer);

    ///read the video frames
    vector<Mat> frames;
    Mat frame,norm_frame;
    if(!video.isOpened())
        exit(1);
    else{
        while(video.isOpened()){
            video >> frame;
            if(frame.empty())
                break;
            cvtColor(frame,frame,CV_BGR2GRAY);
            resize(frame,frame,Size(160,120));
            frame.convertTo(frame,CV_32FC1);
            normalize(frame,frame,0.0,1.0,NORM_MINMAX,-1);
            frames.push_back(frame);
        }
    }
    ///create the blur vector
    Mat tempBlur;
    vector<Mat> videoBlur;
    for(int i=0;i<frames.size();i++){
        tempBlur=conv2(frames[i],blurKer);
        divide(tempBlur,mask,tempBlur,1,CV_32FC1);
        videoBlur.push_back(tempBlur);
    }
    ///make the gradient vector
    vector<Mat> grad_frames;
    Mat res_diff;
    Mat res_temp;
    for(int i=0;i<frames.size()-1;i++){

        res_diff=Mat::zeros(120,160,CV_32FC1);
        subtract(videoBlur[i],videoBlur[i+1],res_diff,Mat(),CV_32FC1);
        res_diff=abs(res_diff);
        res_temp=res_diff.clone();
        grad_frames.push_back(res_temp);
    }
    //cout << grad_frames[0] << endl;
    int counter=0;

    ///motionReg computation
    Mat motionReg[grad_frames.size()];
    Mat temp=Mat::zeros(patchWin,patchWin,CV_32FC1);
    //Mat temp2=Mat::zeros(patchWin,patchWin,CV_32FC1);
    Mat motionResponse=Mat::zeros(12,16,CV_32FC1);
    //vector<Mat> m2;
    //Mat kernel=Mat::ones(patchWin,patchWin,CV_32FC1);
    //Mat tmpMotion;
    //Mat tmpSum = Mat::zeros(12,16,CV_32FC1);

    for(int i=0;i<grad_frames.size();i++){
        //motionResponse[i]=Mat::zeros(12,16,CV_32FC1);//.push_back(temp);
        motionReg[i]=Mat::zeros(12,16,CV_32FC1);//push_back(temp);
    }
    //initialize motionReg
    vector<Mat> temp_vec;
    for(int i=1;i<=BKH;i++){
        for(int j=1;j<=BKW;j++){
            for(int k=0;k<grad_frames.size();k++){
                temp=grad_frames[k].rowRange(patchWin*(i-1),patchWin*i).colRange(patchWin*(j-1),patchWin*j);
                motionReg[k].at<float>(i-1,j-1)=sum(temp.clone())[0];
            }
        }
    }
    cout << "reached here" << endl;
    int length=tprLen*pow(patchWin,2);
    vector<vector<float> > features;  //features
    //vector<Mat> location;  //feature locations
    counter=0;
    vector<Mat> cube;
    Mat cube_tmp(patchWin,patchWin,CV_32FC1);
    vector<float> featureTmp;
    vector<int> tmp_loc;
    cout << grad_frames.size() << endl;
    for(int frameID=tprLen; frameID <= grad_frames.size()-tprLen-1; frameID++){
        motionResponse=Mat::zeros(12,16,CV_32FC1);
        for(int ii=frameID-2;ii<=frameID+2;ii++){
            motionResponse=motionResponse+motionReg[ii].clone();
        }
        for(int ii=1 ; ii <= BKH ; ii++){

            for(int jj=1 ; jj <= BKW ; jj++){
                //cout << motionResponse[frameID].at<float>(ii,jj) << endl;
                if(motionResponse.at<float>(ii,jj) > MT_thr){
                    //cout <<"r" << endl;
                    counter+=1;
                    for(int kk=frameID-2;kk<=frameID+2;kk++){

                        cube_tmp=grad_frames[kk].rowRange(patchWin*(ii-1),ii*patchWin).colRange((jj-1)*patchWin,jj*patchWin);
                        //cout << cube_tmp.at<float>(0,0) << endl;
                        //feature vector assign
                        for(int k=0;k<cube_tmp.rows;k++){
                            for(int l=0;l<cube_tmp.cols;l++){
                                featureTmp.push_back(cube_tmp.at<float>(k,l));
                            }
                        }
                    }
                    //cout << "cube found" << endl;
                    features.push_back(featureTmp);
                    tmp_loc.push_back(ii-1);
                    tmp_loc.push_back(jj-1);
                    tmp_loc.push_back(frameID);
                    location.push_back(tmp_loc);

                    tmp_loc.clear();
                    //cout << features.size() << endl;
                    featureTmp.clear();
                }
            }
        }
    }
    cout << counter << endl;
    cout << "reached the end" << endl;
    return features;
}
