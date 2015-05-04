/*
 * PhotometricStereoSolver.h
 *
 *  Created on: Mar 16, 2015
 *      Author: vincy
 */

#ifndef SRC_PHOTOMETRICSTEREOSOLVER_H_
#define SRC_PHOTOMETRICSTEREOSOLVER_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

namespace ps {

class PhotometricStereoSolver {
public:
	PhotometricStereoSolver();
	PhotometricStereoSolver(unsigned numImages):numImages(numImages){};
	virtual ~PhotometricStereoSolver();

public:
	int processInputImages(cv::Mat* shadowMask, const cv::Mat* srcImages, float shadowThreshold = 0.1);
	int applyPhotometricStereo(const cv::Mat* shadowMask, const cv::Mat* srcImages, const cv::Vec3f* srcLight, cv::Mat& albedo, cv::Mat& norm);
	int dumpResults(string format,const string filename, const cv::Mat& depthMap);
	int depthFromGradient( const cv::Mat norm, cv::Mat& depthMap, int generateMode = 0);

	int applyNearRangePS(cv::Mat& depthMap, const cv::Mat* shadowMask, const cv::Mat* srcImages, const cv::Vec3f* srcLight, cv::Mat& albedo, cv::Mat& norm, int Max_iter);

private:
	cv::Mat getGlobalHeights(cv::Mat Pgrads, cv::Mat Qgrads);
	cv::Mat performFCAlgo(const cv::Mat&Pmat, const cv::Mat&Qmat, float labmda = 0.5);
	void Vintegrate( Mat &P,  Mat &Q,  Mat&Z, int width, int height, float mu = 0.5);
	void integrate( float *P,  float *Q,  float *Z, int width, int height, float mu = 0.5);
	int numImages;

};

} /* namespace std */

#endif /* SRC_PHOTOMETRICSTEREOSOLVER_H_ */
