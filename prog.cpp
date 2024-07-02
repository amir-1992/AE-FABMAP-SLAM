#include <iostream>
#include <string>
#include "opencv2/core.hpp"
#include <fstream>
using namespace cv;
using namespace std;

int main (int, char** argv){
        //create writer
        FileStorage fs("/home/amir/Desktop/ultram/exp2/394_BOW.yml", FileStorage::WRITE);
	cv::Mat depthImg;
	int rows = 0;
        std::ifstream file("/home/amir/Desktop/ultram/exp2/394_BOW.txt");
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
	fs << "vocab" << vocab;

        //release file
        fs.release();
        return 0;

}
