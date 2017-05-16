/*

Author:Kushagra Pandey
Project title: IKST- Abnormal event detection at 150FPS in Matlab
Institute: Indian Institute of Technology,Bhubaneswar
Date Modified:02/02/2016
Supervisor: Dr. D.P. Dogra

=================Algorithm for Training================

The training procedure is as follows:-
*/
#include "trainSystem.h"
#include "featureExtractor.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

trainSystem::trainSystem()
{
    //constructor
}
double trainSystem::calcNorm(Mat inp){
        double val=0;
        for(int i=0;i<inp.rows;i++){
            for(int j=0;j<inp.cols;j++){
                val+=(inp.at<float>(i,j)*inp.at<float>(i,j));
            }
        }
        return sqrt(val);
}
/*double trainSystem::accuracy(int length){
        int val=0;
        if(length<=500)
            val=0.3;
        else if(length <= 1000)
            val=1;
        else if(length <= 5000)
            val=3;
        else if(length <= 10000)
            val=6;
        else if(length <= 20000)
            val=15;
        else if(length <= 50000)
            val=20;
        else
            val=50;
        return val;
}*/
Mat trainSystem::findSCombination(Mat currentFeatures,int numCenters,vector<int> &gamma){

        int windowSize=5;

        //store the sparse combination set, beta and the gamma parameters
        Mat SparseSet(numCenters,currentFeatures.rows,CV_32F);
        vector<Mat> beta; //stores the beta parameter for all the elements of the combination set

        //perform k means clustering for initial initialization of the system
        Mat bestlabels;
        int attempts=5;
        kmeans(currentFeatures.t(),numCenters,bestlabels,TermCriteria(CV_TERMCRIT_EPS,10000,0.001),attempts,KMEANS_PP_CENTERS,SparseSet);
        SparseSet=SparseSet.t();
        //cout << SparseSet.size() << endl;

        //these parameters are to be loaded from the file
        float delta=0.003;
        float lambda_error=0.1;

        double train_error=10000; //take a large value initially
        double nextValue,prevValue;
        int counter=0;
        //initializing gamma parameter as all 1's
        gamma.clear();
        for(int i=0;i<currentFeatures.cols;i++)
            gamma.push_back(1);

        double epsilon=0.0001; //epsilon for convergence of the equation
        //apply the optimization algorithm for SparseSet and the beta parameters
        //double t=accuracy(currentFeatures.cols);
        Mat temp_mat=Mat::zeros(numCenters,100,CV_32FC1);
        Mat temp_inv_mat=Mat::zeros(numCenters,100,CV_32FC1);

        while(true){//train_error > accuracy(currentFeatures.cols)){

            train_error=0;
            beta.clear();

            //optimize beta while keeping S constant
            temp_mat=((SparseSet.t() * SparseSet).inv(DECOMP_SVD)) * SparseSet.t();
            for(int i=0;i<currentFeatures.cols;i++)
                beta.push_back(temp_mat * currentFeatures.col(i)) ;

            //optimize S while keeping beta constant and gamma as 1's for all feature vectors
            Mat sum=Mat::zeros(100,numCenters,CV_32FC1);
            for(int i=0;i<currentFeatures.cols;i++){
                sum=sum+2*gamma[i]*(SparseSet*beta[i]-currentFeatures.col(i))*(beta[i].t());
            }
            //cout << sum << endl;
            //while(waitKey(0)!='q'){}
            SparseSet=SparseSet-delta*sum;

            //optimize gamma parameter
            gamma.clear();
            train_error=0;
            for(int i=0;i<currentFeatures.cols;i++){
                train_error=pow(calcNorm(currentFeatures.col(i)-SparseSet*beta[i]),2);
                //cout << train_error << "\t" << endl;
                if(train_error <= lambda_error)
                    gamma.push_back(1);
                else
                    gamma.push_back(0);
            }

            //calculate the L(beta,S) value for the convergence criteria
            train_error=0;
            double val=0;
            for(int i=0;i<currentFeatures.cols;i++){
                temp_mat=currentFeatures.col(i)-SparseSet*beta[i];
                train_error+=gamma[i]*pow(calcNorm(temp_mat),2);
            }
            double mse=train_error/currentFeatures.cols;
            //cout <<"Overall error:" << mse << endl;
            //check if the convergence has been achieved
            if(counter==0){
                nextValue=mse;
                counter+=1;
            }
            else{
                prevValue=nextValue;
                nextValue=mse;
                if(abs(nextValue-prevValue) < epsilon)
                    break;
            }
            //while(waitKey(0)!='q'){}
        }
        cout << "convergence achieved" << endl;
        /*beta.clear();
        //update beta for the last iteration
        temp_mat=((SparseSet.t() * SparseSet).inv(DECOMP_SVD))* SparseSet.t();
        for(int i=0;i<currentFeatures.cols;i++)
            beta.push_back(temp_mat * currentFeatures.col(i));
        //optimize gamma parameter
        gamma.clear();
        train_error=0;
        for(int i=0;i<currentFeatures.cols;i++){
            train_error=pow(calcNorm(currentFeatures.col(i)-SparseSet*beta[i]),2);
            //cout << train_error << "\t" << endl;
            if(train_error <= lambda_error)
                gamma.push_back(1);
            else
                gamma.push_back(0);
        }*/
        //cout << Mat(gamma).t() << endl;
        //while(waitKey(0)!='q'){}
        return SparseSet;
}
vector<Mat> trainSystem::train(vector<vector<float> > regionFeatures){

        //training parameters
        int windowSize=5;
        int sparsity=20; //set it later to the experimental value
        //stores all the sparse combination sets
        vector<Mat> sparseCombSet;
        Mat sparseCombination;  //used to store a single sparse combination
        vector<int> K;
        int k_temp=0;
        vector<int> gamma;


        int non_zero_index=0,counter=0;
        //find the sparse combination sets for the region features
        while(regionFeatures.size()!=0){

            cout << "--------------Combination Set learning started------------" << endl;
            counter=0;
            non_zero_index=0;
            //convert the regionFeatures vector to Mat format
            Mat regionFeatures_M(regionFeatures[0].size(),regionFeatures.size(),CV_32FC1);
            for(int j=0;j<regionFeatures_M.cols;j++){
                for(int k=0;k<regionFeatures_M.rows;k++){
                    regionFeatures_M.at<float>(k,j)=(regionFeatures.at(j)).at(k);
                }
            }
            //find the k_temp'th sparse combination
            sparseCombination=findSCombination(regionFeatures_M,sparsity,gamma);
            //cout << "The sparse combination has been found" <<endl;

            sparseCombSet.push_back(sparseCombination);
            //remove the features from the region feature space for which gamma parameter is one
            vector<vector<float> > rem_features;
            for(int j=0;j<gamma.size();j++){
                if(gamma[j]==0)
                    rem_features.push_back(regionFeatures[j]);
            }
            if(rem_features.size()==regionFeatures.size())
                break;
            regionFeatures.clear();
            for(int j=0;j<rem_features.size();j++)
                regionFeatures.push_back(rem_features[j]);
            cout << "Remaining features:" << regionFeatures.size()<< endl;
            rem_features.clear();
            k_temp+=1;
            //remove the features if the number of features are less than the sparsity
            if(regionFeatures.size() < sparsity)
                regionFeatures.clear();
        }
        cout << "Sparse combination set has been found" <<endl;
        cout << "Number of combinations:" << k_temp << endl;
        K.push_back(k_temp);
        k_temp=0;
        return sparseCombSet;
}
