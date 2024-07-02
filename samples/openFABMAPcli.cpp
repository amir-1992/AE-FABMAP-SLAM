/*//////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// This file originates from the openFABMAP project:
// [http://code.google.com/p/openfabmap/] -or-
// [https://github.com/arrenglover/openfabmap]
//
// For published work which uses all or part of OpenFABMAP, please cite:
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6224843]
//
// Original Algorithm by Mark Cummins and Paul Newman:
// [http://ijr.sagepub.com/content/27/6/647.short]
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5613942]
// [http://ijr.sagepub.com/content/30/9/1100.abstract]
//
//                           License Agreement
//
// Copyright (C) 2012 Arren Glover [aj.glover@qut.edu.au] and
//                    Will Maddern [w.maddern@qut.edu.au], all rights reserved.
//
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistribution's of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistribution's in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * The name of the copyright holders may not be used to endorse or promote
//    products derived from this software without specific prior written
///   permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability,or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//////////////////////////////////////////////////////////////////////////////*/

#include <openfabmap.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>

#if CV_MAJOR_VERSION == 2 and CV_MINOR_VERSION == 3

#elif CV_MAJOR_VERSION == 2 and CV_MINOR_VERSION == 4
#if USENONFREE
#include <opencv2/nonfree/nonfree.hpp>
#endif
#elif CV_MAJOR_VERSION >= 3
#ifdef USENONFREE
#include <opencv2/xfeatures2d.hpp>
#endif

#if CV_MAJOR_VERSION >= 4
#define CV_CAP_PROP_POS_FRAMES cv::CAP_PROP_POS_FRAMES
#define CV_CAP_PROP_FRAME_COUNT cv::CAP_PROP_FRAME_COUNT
#define CV_AA cv::LINE_AA
#endif
#endif

#include <fstream>
#include <iostream>
#include <string> // but you do use std::string
#include <stdio.h>
#include <string>  
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include<algorithm>
#include <opencv2/features2d.hpp>
#include<iostream>
#include<vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
/*
openFABMAP procedural functions
*/

int showFeatures(std::string trainPath,
                 cv::Ptr<cv::FeatureDetector> &detector);
int generateVocabTrainData(std::string trainPath,
                           std::string vocabTrainDataPath,
                           cv::Ptr<cv::FeatureDetector> &detector,
                           cv::Ptr<cv::DescriptorExtractor> &extractor);
                           
int trainVocabulary(std::string vocabPath,
                    std::string vocabTrainDataPath,
                    double clusterRadius);

int generateBOWImageDescs(std::string dataPath,
                          std::string bowImageDescPath,
                          std::string vocabPath,
                          cv::Ptr<cv::FeatureDetector> &detector,
                          cv::Ptr<cv::DescriptorExtractor> &extractor,
                          int minWords);

int trainChowLiuTree(std::string chowliutreePath,
                     std::string fabmapTrainDataPath,
                     double lowerInformationBound);

int openFABMAP(std::string testPath,
               of2::FabMap *openFABMAP,
               std::string vocabPath,
               std::string resultsPath,
               bool addNewOnly);

int openFABMAP(std::string testPath,
               of2::FabMap *openFABMAP,
               std::string vocabPath,
               std::string resultsPath,
               bool addNewOnly);

/*
helper functions
*/
of2::FabMap *generateFABMAPInstance(cv::FileStorage &settings);
cv::Ptr<cv::FeatureDetector> generateDetector(cv::FileStorage &fs);
cv::Ptr<cv::DescriptorExtractor> generateExtractor(cv::FileStorage &fs);

/*
Advanced tools for keypoint manipulation. These tools are not currently in the
functional code but are available for use if desired.
*/
void drawRichKeypoints(const cv::Mat& src, std::vector<cv::KeyPoint>& kpts,
                       cv::Mat& dst);
void filterKeypoints(std::vector<cv::KeyPoint>& kpts, int maxSize = 0,
                     int maxFeatures = 0);
void sortKeypoints(std::vector<cv::KeyPoint>& keypoints);


/*
The openFabMapcli accepts a YML settings file, an example of which is provided.
Modify options in the settings file for desired operation
*/
int main(int argc, char * argv[])
{
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //load the settings file
    std::string settfilename;
    if (argc == 1) {
        //assume settings in working directory
        settfilename = "settings.yml";
    } else if (argc == 3) {
        if(std::string(argv[1]) != "-s") {
            //incorrect option
            return 0;
        } else {
            //settings provided as argument
            settfilename = std::string(argv[2]);
        }
    } else {
        //incorrect arguments
        return 0;
    }	
    cv::FileStorage fs;
    fs.open(settfilename, cv::FileStorage::READ);
 	std::cout<<settfilename;   
    if (!fs.isOpened()) {
        std::cerr << "Could not open settings file: " << settfilename <<
                     std::endl;
        return -1;
    }
    
    cv::Ptr<cv::FeatureDetector> detector = generateDetector(fs);
    if(!detector) {
        std::cerr << "Feature Detector error" << std::endl;
        return -1;
    }
    cv::Ptr<cv::DescriptorExtractor> extractor = generateExtractor(fs);
    if(!extractor) {
        std::cerr << "Feature Extractor error" << std::endl;
        return -1;
    }
    
    using std::string; // explicit using
    //showFeatures(fs["FilePaths"]["TrainPath"], detector);
    //run desired function
    int result = 0;
    std::string function = "RunOpenFABMAP";
    //function ;
    if (function == "generateBOWImageDescs") {
        result = generateVocabTrainData(fs["FilePaths"]["TrainPath"],
                fs["FilePaths"]["TrainFeatDesc"],
                detector, extractor);

    } else if (function == "TrainVocabulary") {
        result = trainVocabulary(fs["FilePaths"]["Vocabulary"],
                fs["FilePaths"]["TrainFeatDesc"],
                fs["VocabTrainOptions"]["ClusterSize"]);
    }
    else if (function == "GenerateFABMAPTrainData") {
        result = generateBOWImageDescs(fs["FilePaths"]["TrainPath"],
                fs["FilePaths"]["TrainImagDesc"],
                fs["FilePaths"]["Vocabulary"], detector, extractor,
                fs["BOWOptions"]["MinWords"]);
    } else if (function == "TrainChowLiuTree") {
        result = trainChowLiuTree(fs["FilePaths"]["ChowLiuTree"],
                fs["FilePaths"]["TrainImagDesc"],
                fs["ChowLiuOptions"]["LowerInfoBound"]);
    } else if (function == "GenerateFABMAPTestData") {
        result = generateBOWImageDescs(fs["FilePaths"]["TestPath"],
                fs["FilePaths"]["TestImageDesc"],
                fs["FilePaths"]["Vocabulary"], detector, extractor,
                fs["BOWOptions"]["MinWords"]);
    }
    else if (function == "RunOpenFABMAP") {
        std::string placeAddOption = fs["FabMapPlaceAddition"];
        bool addNewOnly = (placeAddOption == "NewMaximumOnly");
        of2::FabMap *fabmap = generateFABMAPInstance(fs);
        if(fabmap) {
            result = openFABMAP(fs["FilePaths"]["TestImageDesc"], fabmap,
                    fs["FilePaths"]["Vocabulary"],
                    fs["FilePaths"]["FabMapResults"], addNewOnly);
        }

    } else {
        std::cerr << "Incorrect Function Type" << std::endl;
        result = -1;
    }
    std::cout << "openFABMAP done" << std::endl;
    std::cin.sync(); std::cin.ignore();
    fs.release();
    return result;
}

int showFeatures(std::string trainPath, cv::Ptr<cv::FeatureDetector> &detector)
{
    cv::VideoCapture movie;
    movie.open(trainPath);
    if (!movie.isOpened()) {
        std::cerr << trainPath << ": training movie not found" << std::endl;
        return -1;
    }
    std::cout << "Press Esc to Exit" << std::endl;
    cv::Mat frame, kptsImg;
    movie.read(frame);
    std::vector<cv::KeyPoint> kpts;
    while (movie.read(frame)) {
        std::cout << typeid(frame).name() << '\n';
    
        detector->detect(frame, kpts);

        std::cout << kpts.size() << " keypoints detected...         \r";
        fflush(stdout);

        cv::drawKeypoints(frame, kpts, kptsImg);

        cv::imshow("Features", kptsImg);
        if(cv::waitKey(5) == 27) {
            break;
        }
    }
    std::cout << std::endl;
    cv::destroyWindow("Features");
    return 0;
}
 int  getFilePaths(vector<string> &filepaths, cv::String filePath)
{
	filepaths.clear();
	cout << "Read files from: " << filePath << endl;
	vector<cv::String> fn;
	cv::glob(filePath, fn, false);
 
	//prepare pair for sort 
	vector<pair<int, string>> v1;
	pair<int, string> p1;
	vector<cv::String >::iterator it_;
	for (it_ = fn.begin(); it_ != fn.end();	++it_)
	{	
		 //1. Get the file name without path, 1.txt
		string::size_type iPos = (*it_).find_last_of('/') + 1;
		string filename = (*it_).substr(iPos, (*it_).length() - iPos);
		 //2. Get the file name without suffix, 1
	    string name = filename.substr(0, filename.rfind("."));
		 //3. Construct a pair of keys and values
		try {
			 //Prevent errors caused by non-integer file names in the folder
			p1 = make_pair(stoi(name), (*it_).c_str());
 
		}catch(exception e)
		{
			cout << "Crushed -> " << e.what() << endl;
			 //continue; Same as continue directly 
			it_ = fn.erase(it_);
			//https://www.cnblogs.com/shiliuxinya/p/12219149.html
			 it_--; 
		}
		v1.emplace_back(p1);
	}
	//cout << "v1.sie(): " << v1.size()<<endl;
	sort(v1.begin(), v1.end(), [](auto a, auto b) {return a.first < b.first; });
	vector<pair<int, string> >::iterator it;
	for (it = v1.begin(); it != v1.end(); ++it)
	{
		//cout << it->first << endl;
		//cout << it->second << endl;
 
		filepaths.emplace_back(it->second);
	}
	return 0;
}


int generateVocabTrainData(std::string trainPath,
                           std::string vocabTrainDataPath,
                           cv::Ptr<cv::FeatureDetector> &detector,
                           cv::Ptr<cv::DescriptorExtractor> &extractor)
{	 
	std::cout<<vocabTrainDataPath;
	cv::Mat img; 
	vector<string> filePaths;
	cv::String folderPath = "/home/amir/Desktop/newcollege/test/";
	//getFilePaths(filePaths, folderPath);
	//////////////////"RunOpenFABMAP"///////////////////////////////////////////////////////////////
	std::vector<cv::Mat> images;
	getFilePaths(filePaths, folderPath);
    cv::Mat vocabTrainData;
    cv::Mat frame, descs, feats;
    std::vector<cv::KeyPoint> kpts;
    std::cout.setf(std::ios_base::fixed);
    std::cout.precision(0);
    for (size_t i = 1; i <filePaths.size(); ++i)
    {  
    	img = cv::imread(filePaths[i]);
    	std::cout<<filePaths[i]<<std::endl;
		cvtColor(img, img, cv::COLOR_BGR2GRAY);
	   //detect & extract features
      detector->detect(img, kpts);
      extractor->compute(img, kpts, descs);
     //add all descriptors to the training data
     vocabTrainData.push_back(descs);
     //show progress
     cv::drawKeypoints(img, kpts, feats);
     cv::imshow("Training Data", feats);
     fflush(stdout);

     if(cv::waitKey(5) == 27) {
         cv::destroyWindow("Training Data");
         std::cout << std::endl;
         return -1;
     }

    }
    cv::destroyWindow("Training Data");
    std::cout << "Done: " << vocabTrainData.rows << " Descriptors" << std::endl;
    cv::FileStorage fs;
    fs.open(vocabTrainDataPath, cv::FileStorage::WRITE);
    fs << "VocabTrainData" << vocabTrainData;
    fs.release();
    return 0;
}

int trainVocabulary(std::string vocabPath,
                    std::string vocabTrainDataPath,
                    double clusterRadius)
{
    //ensure not overwriting a vocabulary
    std::ifstream checker;
    checker.open(vocabPath.c_str());
    if(checker.is_open()) {
        std::cerr << vocabPath << ": Vocabulary already present" <<
                     std::endl;
        checker.close();
        return -1;
    }

    std::cout << "Loading vocabulary training data" << std::endl;

    cv::FileStorage fs;

    //load in vocab training data
    fs.open(vocabTrainDataPath, cv::FileStorage::READ);
    cv::Mat vocabTrainData;
    fs["VocabTrainData"] >> vocabTrainData;
    if (vocabTrainData.empty()) {
        std::cerr << vocabTrainDataPath << ": Training Data not found" <<
                     std::endl;
        return -1;
    }
    fs.release();
    std::cout << "Performing clustering" << std::endl;
    //uses Modified Sequential Clustering to train a vocabulary
    of2::BOWMSCTrainer trainer(clusterRadius);
    trainer.add(vocabTrainData);
    cv::Mat vocab = trainer.cluster();

    //save the vocabulary
    std::cout << "Saving vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::WRITE);
    fs << "Vocabulary" << vocab;
    fs.release();
    return 0;
}
 
int generateBOWImageDescs(std::string dataPath,
                          std::string bowImageDescPath,
                          std::string vocabPath,
                          cv::Ptr<cv::FeatureDetector> &detector1,
                          cv::Ptr<cv::DescriptorExtractor> &extractor1,
                          int minWords)
{       
	std::cout<<"here";
	
	cv::Mat depthImg;
	int rows = 0;
        std::ifstream file("/home/amir/Desktop/newcollege/CENTROID_TEST_OPENFABMAP.yml");	
	std::string line;
	while (std::getline(file, line)) {  
	    std::istringstream stream(line);
	    char sep; 
	    double x;
	    // read *both* a number and a comma:
	    do{

		depthImg.push_back(x);
	    }while(stream >> x && stream >> sep );
	    rows ++;
	} 

    // reshape to 2d:
    cv::Mat vocab = depthImg.reshape(1,rows);
    //  = depthImg; 
	//////////////////////////////////////////////////////////
	
    //std::cout << "image1 row: 0~2 = "<< std::endl << " "  << vocab.rowRange(0, 1) << std::endl << std::endl;
    //cv::Ptr<cv::DescriptorExtractor> extractor =  
/////////////////////////////////////////////////////////////////////////////////
	vector<string> filePaths;
	cv::String folderPath = "/home/amir/Desktop/openfabmap_sample/stlucia_train/";
	//getFilePaths(filePaths, folderPath);
/////////////////////////////////////////////////////////////////////////////////
	std::vector<cv::Mat> images;
 	getFilePaths(filePaths, folderPath);
   //glob(path, fn, false);
   
   //cv::String folder = "/home/amir/Downloads/openfabmap-master/rgb1/*.png";
   //std::vector<cv::String> filenames;
   //cv::glob(folder, filenames);
   // here is the problem


	int i =0;
	cv::Mat graymat1 = cv::imread(filePaths[0],cv::IMREAD_GRAYSCALE);	 
	cv::Mat graymat=graymat1;		 
	
    	
   	
   //create a nearest neighbor matcher
    cv::Ptr<cv::DescriptorMatcher> matcher;

matcher = cv::BFMatcher::create();
    //create Sift feature point extracter
    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create(500);
   //Ptr< cv::xfeatures2d::SIFT> detector =  xfeatures2d::SIFT::create();
    /////////////////////////////////////////////////////////////////////////
    //create Sift descriptor extractor
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create(500);
    //Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor());    
    	std::cout<<"4"<<std::endl;
  	cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);
	std::cout<<"5"<<std::endl;
	cv::Mat floatdictionary; 
	floatdictionary = vocab;
	
	cv::Mat udictionary = floatdictionary;  
	udictionary.convertTo(udictionary,CV_32F); 
	bowDE.setVocabulary(udictionary); 
		 	std::cout<<"6"<<std::endl;
	
	vector<KeyPoint> keypoints;        
    //Detect SIFT keypoints (or feature points)
    detector->detect(graymat,keypoints);
    std::cout<<"7"<<std::endl;
    std::cout<<keypoints.size()<<std::endl;
    //To store the BoW (or BoF) representation of the image
    Mat bowDescriptor;   
   //bowDescriptor.convertTo(bowDescriptor,CV_32F);      
    //extract BoW (or BoF) descriptor from given image
    bowDE.compute(graymat,keypoints,bowDescriptor);
	std::cout<<"8"<<std::endl;

    
    std::ofstream maskw;	
    cv::Mat img;  
	cv::Mat fabmapTrainData;
    std::string bowImageDescPath1 = "/home/amir/Desktop/openfabmap_sample/exiso_train.yml";
    if(minWords)
    {     //std::cout<<"1";
        maskw.open(std::string(bowImageDescPath1 + "mask.txt").c_str());    
    		 //std::cout<<"2";
    }
    
    for (size_t i = 1; i <filePaths.size(); ++i)
    {  
    	img = cv::imread(filePaths[i],cv::IMREAD_GRAYSCALE);
    	std::cout<<i<<std::endl;
    	//std::cout<<bowDescriptor.size()<<std::endl;
	//cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	//img.convertTo(img, CV_8U);
	//
	//std::cout<<"4";

	detector->detect(img, keypoints);
	//bowDescriptor.convertTo(bowDescriptor,CV_32F);
	bowDE.compute(img, keypoints, bowDescriptor);
        //std::cout<<keypoints.size()<<endl;
	
        if(minWords) {
            //writing a mask file
            if(cv::countNonZero(bowDescriptor) < minWords) {
                //frame masked
                maskw << "0" << std::endl;
            } else {
                //frame accepted
                maskw << "1" << std::endl;
                fabmapTrainData.push_back(bowDescriptor);
            }
        } else {
            fabmapTrainData.push_back(bowDescriptor);
        }
    }
    cv::FileStorage fs("/home/amir/Desktop/openfabmap_sample/exiso_train.yml", cv::FileStorage::WRITE);
    fs << "BOWImageDescs" << fabmapTrainData;
    //fs.open(bowImageDescPath1, cv::FileStorage::WRITE);
    fs.release();   
    std::cout << "Done " << std::endl; 
    return 0;
}

////////////////////////////////////////////////////////////////////////
// Names are arranged in ascending order, source blog: https://blog.csdn.net/sss_369/article/details/87740843
/////////////////////////////////////////////////////////////////////////////


int trainChowLiuTree(std::string chowliutreePath,
                     std::string fabmapTrainDataPath,
                     double lowerInformationBound)
{
	 //fabmapTrainDataPath = ;
    fabmapTrainDataPath = "/paper/opena2/opena/one/exiso_train.yml";
    cv::FileStorage fs;
    //ensure not overwriting training data
    //load FabMap training data
    fs.open(fabmapTrainDataPath, cv::FileStorage::READ);
    cv::Mat fabmapTrainData;
    fs["BOWImageDescs"] >> fabmapTrainData;
    if (fabmapTrainData.empty()) {
        std::cerr << fabmapTrainDataPath << ": FabMap Training Data not found"
                  << std::endl;
        return -1;
    }
    fs.release();

    //generate the tree from the data
    std::cout << "Making Chow-Liu Tree" << std::endl;
    of2::ChowLiuTree tree;
    tree.add(fabmapTrainData);
    cv::Mat clTree = tree.make(lowerInformationBound);
	 
    //save the resulting tree
    chowliutreePath = "/paper/opena2/opena/one/chowliu_FABMAP.yml";
    std::cout <<"Saving Chow-Liu Tree" << std::endl;
    fs.open(chowliutreePath, cv::FileStorage::WRITE);
    fs << "ChowLiuTree" << clTree;
    fs.release();

    return 0;

}


/*
Run FabMap on a test dataset
*/
int openFABMAP(std::string testPath,
               of2::FabMap *fabmap,
               std::string vocabPath,
               std::string resultsPath,
               bool addNewOnly)
{

    cv::FileStorage fs;

    //ensure not overwritincentroid_train_data200.ymlg results
    std::ifstream checker;
    checker.open(resultsPath.c_str());
    if(checker.is_open()) {
        std::cerr << resultsPath << ": Results already present" << std::endl;
        checker.close();
        return -1;
    }

    //load the vocabulary
    std::cout << "Loading Vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::READ);
    cv::Mat vocab;
    fs["vocab"] >> vocab;
    if (vocab.empty()) {
        std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
        return -1;
    }
    fs.release();

    //load the test"ChowLiuTree" data
    fs.open(testPath, cv::FileStorage::READ);
    cv::Mat testImageDescs;
    fs["vocab"] >> testImageDescs;
    if(testImageDescs.empty()) {
        std::cerr << testPath << ": Test data not found" << std::endl;
        return -1;
    }
    fs.release();

    //running openFABMAP
    std::cout << "Running openFABMAP" << std::endl;
    std::vector<of2::IMatch> matches;
    std::vector<of2::IMatch>::iterator l;



    cv::Mat confusion_mat(testImageDescs.rows, testImageDescs.rows, CV_64FC1);
    confusion_mat.setTo(0); // init to 0's
	
    //new part
    //for(int i = 0; i < testImageDescs.rows; i++) {			    
     std::cout<<": Test data not found" << std::endl;
    if (!addNewOnly) {

        //automatically comparing a whole dataset
        fabmap->localize(testImageDescs, matches, true);

        for(l = matches.begin(); l != matches.end(); l++) {
            if(l->imgIdx < 0) {
                confusion_mat.at<double>(l->queryIdx, l->queryIdx) = l->match;

            } else {
                confusion_mat.at<double>(l->queryIdx, l->imgIdx) = l->match;
            }
        }

    } else {

        //criteria for adding locations used
        for(int i = 0; i < testImageDescs.rows; i++) {
            matches.clear();
            //compare imaVocabges individually
            fabmap->localize(testImageDescs.row(i), matches);

            bool new_place_max = true;
            for(l = matches.begin(); l != matches.end(); l++) {

                if(l->imgIdx < 0) {
                    //add the new place to the confusion matrix 'diagonal'
                    confusion_mat.at<double>(i, (int)matches.size()-1) = l->match;

                } else {
                    //add the score to the confusion matrix
                    confusion_mat.at<double>(i, l->imgIdx) = l->match;
                }

                //test for new location maximum
                if(l->match > matches.front().match) {
                    new_place_max = false;
                }
            }

            if(new_place_max) {
                fabmap->add(testImageDescs.row(i));
            }
        }
    }

    //save the result as plain text for ease of import to Matlab
    std::ofstream writer(resultsPath.c_str());
    for(int i = 0; i < confusion_mat.rows; i++) {
        for(int j = 0; j < confusion_mat.cols; j++) {
            writer << confusion_mat.at<double>(i, j) << " ";
        }
        writer << std::endl;
    }
    writer.close();
    return 0;
}

cv::Ptr<cv::FeatureDetector> generateDetector(cv::FileStorage &fs) {

    //create common feature detector and descriptor extractor
    return cv::xfeatures2d::SURF::create(5000,4,2,true,true);//return the nullptr

}

cv::Ptr<cv::DescriptorExtractor> generateExtractor(cv::FileStorage &fs)
{
   return cv::xfeatures2d::SURF::create(5000,4,2,true,true);
}





/*
create an instance of a FabMap class with the options given in the settings file
*/
of2::FabMap *generateFABMAPInstance(cv::FileStorage &settings)
{

    cv::FileStorage fs;

    //load FabMap training data
    std::string fabmapTrainDataPath = settings["FilePaths"]["TrainImagDesc"];
    std::string chowliutreePath = settings["FilePaths"]["ChowLiuTree"];

    std::cout << "Loading FabMap Training Data" << std::endl;
    fs.open(fabmapTrainDataPath, cv::FileStorage::READ);
    cv::Mat fabmapTrainData;
    fs["vocab"] >> fabmapTrainData;
    if (fabmapTrainData.empty()) {
        std::cerr << fabmapTrainDataPath << ": FabMap Training Data not found"
                  << std::endl;
        return NULL;
    }
    fs.release();

    //load a chow-liu tree
    std::cout << "Loading Chow-Liu Tree" << std::endl;
    fs.open(chowliutreePath, cv::FileStorage::READ);
    cv::Mat clTree;
    fs["ChowLiuTree"] >> clTree;
    if (clTree.empty()) {
        std::cerr << chowliutreePath << ": Chow-Liu tree not found" <<
                     std::endl;
        return NULL;
    }
    fs.release();

    //create options flags
    std::string newPlaceMethod =
            settings["openFabMapOptions"]["NewPlaceMethod"];
    std::string bayesMethod = settings["openFabMapOptions"]["BayesMethod"];
    int simpleMotionModel = settings["openFabMapOptions"]["SimpleMotion"];
    int options = 0;
    if(newPlaceMethod == "Sampled") {
        options |= of2::FabMap::SAMPLED;
    } else {
        options |= of2::FabMap::MEAN_FIELD;
    }
    if(bayesMethod == "ChowLiu") {
        options |= of2::FabMap::CHOW_LIU;
    } else {
        options |= of2::FabMap::NAIVE_BAYES;
    }
    if(simpleMotionModel) {
        options |= of2::FabMap::MOTION_MODEL;
    }

    of2::FabMap *fabmap;

    //create an instance of the desired type of FabMap
    std::string fabMapVersion = settings["openFabMapOptions"]["FabMapVersion"];
    if(fabMapVersion == "FABMAP1") {
        fabmap = new of2::FabMap1(clTree,
                                  settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options,
                settings["openFabMapOptions"]["NumSamples"]);
    } else if(fabMapVersion == "FABMAPLUT") {
        fabmap = new of2::FabMapLUT(clTree,
                                    settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options,
                settings["openFabMapOptions"]["NumSamples"],
                settings["openFabMapOptions"]["FabMapLUT"]["Precision"]);
    } else if(fabMapVersion == "FABMAPFBO") {
        fabmap = new of2::FabMapFBO(clTree,
                                    settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options,
                settings["openFabMapOptions"]["NumSamples"],
                settings["openFabMapOptions"]["FabMapFBO"]["RejectionThreshold"],
                settings["openFabMapOptions"]["FabMapFBO"]["PsGd"],
                settings["openFabMapOptions"]["FabMapFBO"]["BisectionStart"],
                settings["openFabMapOptions"]["FabMapFBO"]["BisectionIts"]);
    } else if(fabMapVersion == "FABMAP2") {
    	       std::cout <<"here1"<< std::endl;
        fabmap = new of2::FabMap2(clTree,
                                  settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options);
    } else {
        std::cerr << "Could not identify openFABMAPVersion from settings"
                     " file" << std::endl;
        return NULL;
    }
    std::cout << "HERE" << std::endl;
    //add the training data for use with the sampling method
    fabmap->addTraining(fabmapTrainData);

    return fabmap;

}



/*
draws keypoints to scale with coloring proportional to feature strength
*/
void drawRichKeypoints(const cv::Mat& src, std::vector<cv::KeyPoint>& kpts, cv::Mat& dst) {

    cv::Mat grayFrame;
    cvtColor(src, grayFrame, cv::COLOR_RGB2GRAY);
    cvtColor(grayFrame, dst, cv::COLOR_GRAY2RGB);

    if (kpts.size() == 0) {
        return;
    }

    std::vector<cv::KeyPoint> kpts_cpy, kpts_sorted;

    kpts_cpy.insert(kpts_cpy.end(), kpts.begin(), kpts.end());

    double maxResponse = kpts_cpy.at(0).response;
    double minResponse = kpts_cpy.at(0).response;

    while (kpts_cpy.size() > 0) {

        double maxR = 0.0;
        unsigned int idx = 0;

        for (unsigned int iii = 0; iii < kpts_cpy.size(); iii++) {

            if (kpts_cpy.at(iii).response > maxR) {
                maxR = kpts_cpy.at(iii).response;
                idx = iii;
            }

            if (kpts_cpy.at(iii).response > maxResponse) {
                maxResponse = kpts_cpy.at(iii).response;
            }

            if (kpts_cpy.at(iii).response < minResponse) {
                minResponse = kpts_cpy.at(iii).response;
            }
        }

        kpts_sorted.push_back(kpts_cpy.at(idx));
        kpts_cpy.erase(kpts_cpy.begin() + idx);

    }

    int thickness = 1;
    cv::Point center;
    cv::Scalar colour;
    int red = 0, blue = 0, green = 0;
    int radius;
    double normalizedScore;

    if (minResponse == maxResponse) {
        colour = CV_RGB(255, 0, 0);
    }

    for (int iii = (int)kpts_sorted.size()-1; iii >= 0; iii--) {

        if (minResponse != maxResponse) {
            normalizedScore = pow((kpts_sorted.at(iii).response - minResponse) / (maxResponse - minResponse), 0.25);
            red = int(255.0 * normalizedScore);
            green = int(255.0 - 255.0 * normalizedScore);
            colour = CV_RGB(red, green, blue);
        }

        center = kpts_sorted.at(iii).pt;
        center.x *= 16;
        center.y *= 16;

        radius = (int)(16.0 * ((double)(kpts_sorted.at(iii).size)/2.0));

        if (radius > 0) {
            circle(dst, center, radius, colour, thickness, CV_AA, 4);
        }

    }

}

/*
Removes surplus features and those with invalid size
*/
void filterKeypoints(std::vector<cv::KeyPoint>& kpts, int maxSize, int maxFeatures) {

    if (maxSize == 0) {
        return;
    }

    sortKeypoints(kpts);

    for (unsigned int iii = 0; iii < kpts.size(); iii++) {

        if (kpts.at(iii).size > float(maxSize)) {
            kpts.erase(kpts.begin() + iii);
            iii--;
        }
    }

    if ((maxFeatures != 0) && ((int)kpts.size() > maxFeatures)) {
        kpts.erase(kpts.begin()+maxFeatures, kpts.end());
    }

}

/*
Sorts keypoints in descending order of response (strength)
*/
void sortKeypoints(std::vector<cv::KeyPoint>& keypoints) {

    if (keypoints.size() <= 1) {
        return;
    }

    std::vector<cv::KeyPoint> sortedKeypoints;

    // Add the first one
    sortedKeypoints.push_back(keypoints.at(0));

    for (unsigned int i = 1; i < keypoints.size(); i++) {

        unsigned int j = 0;
        bool hasBeenAdded = false;

        while ((j < sortedKeypoints.size()) && (!hasBeenAdded)) {

            if (abs(keypoints.at(i).response) > abs(sortedKeypoints.at(j).response)) {
                sortedKeypoints.insert(sortedKeypoints.begin() + j, keypoints.at(i));

                hasBeenAdded = true;
            }

            j++;
        }

        if (!hasBeenAdded) {
            sortedKeypoints.push_back(keypoints.at(i));
        }

    }

    keypoints.swap(sortedKeypoints);

}
