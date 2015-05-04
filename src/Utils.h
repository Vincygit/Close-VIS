/*
 * Util.h
 *
 *  Created on: Mar 16, 2015
 *      Author: vincy
 */

#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

static inline void help(char* progName)
{
	cout << endl
			<<  "This is an demo for auto-calibration program of photometric stereo. " << endl
			<< progName << " [image_name] "                       << endl << endl;
}

std::vector<std::string> split(const std::string &s, char delim);
int filterImage(cv::Mat& img, cv::Mat& output, int windowSize=3, float sigma=10);

/**
 * if the fourth parameter is set, then
 * the we will pick height or width that makes this image smallest.
 * */
int rescaleImage(cv::Mat& img, int height, int width, int fixedRatio = 1);
int cropImage(cv::Mat& img, cv::Mat& result, int ratio );

#endif /* SRC_UTILS_H_ */
