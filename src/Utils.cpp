/*
 * Util.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: vincy
 */

#include "Utils.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
/**
 * split stream by using string stream
 * */
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

/**
 * apply split function
 * */
std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

/**
 * preprocess the input image.
 * Low pass filter. de-noise.
 * */
inline int filterImage(cv::Mat& img, cv::Mat& output, int windowSize, float sigma)
{
	cv::GaussianBlur(img,output,cv::Size(windowSize,windowSize),sigma);
	return 1;
}

int rescaleImage(cv::Mat& img, int height, int width, int fixedRatio)
{
	int oriH = img.cols;
	int oriW = img.rows;
	if(fixedRatio) {
		float nRatio = (float)height / (float) width;
		float oRatio = (float)oriH / (float) oriW;

		if(nRatio < oRatio )
		{
			width  = (float)height / oRatio;
		} else if(nRatio > oRatio) {
			height = (float)width * oRatio;
		}
	}

	if(oriH == height && oriW == width)
		return 0;

	if(fixedRatio)
		cout<< "fixed ratio scale, with new height:" <<height <<" and new width:"<< width <<endl;
	else
		cout<< "scale, with new height:" <<height <<" and new width:"<< width <<endl;
//	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//	//	imwrite( string(param.dataFolder + "albedo.jpg"), albedo );
//
//	cv::imshow("Display window", img);
//	cv::waitKey();

	cv::resize(img, img, cv::Size(height, width),0,0,CV_INTER_LINEAR);
	return 1;
}

/**
 * crop the image to the assigned ratio region
 * @param ratio: 4 8
 * image must be 3 channel.
 * */
int cropImage(cv::Mat& img, cv::Mat & result, int ratio )
{
	result.create(cv::Size(2*img.cols/ratio ,2*img.rows/ratio), CV_8UC3);
	for(int h = img.rows/ratio; h < img.rows * (ratio-1)/ratio; h++)
	{
		for(int w = img.cols/ratio; w < img.cols * (ratio-1)/ratio; w++)
		{
			result.at<cv::Vec3b>(h - img.rows/ratio, w - img.cols/ratio)[0] = img.at<cv::Vec3b>(h,w)[0];
			result.at<cv::Vec3b>(h - img.rows/ratio, w - img.cols/ratio)[1] = img.at<cv::Vec3b>(h,w)[1];
			result.at<cv::Vec3b>(h - img.rows/ratio, w - img.cols/ratio)[2] = img.at<cv::Vec3b>(h,w)[2];
		}
	}
	return 1;
}
