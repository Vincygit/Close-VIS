/*
 * Calibrator.h
 *
 *  Created on: Mar 13, 2015
 *      Author: vincy
 */

#ifndef SRC_CALIBRATOR_H_
#define SRC_CALIBRATOR_H_
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include "BestFit.h"
#include "Utils.h"

using namespace cv;

namespace ps {

#define MSE
#define USE_WATERSHED
#define FittedCircleThickness 3

// deprecated.
#define HIGHLIGHT_COUNT 500
#define HIGHLIGHT_RATIO 0.5

/**
 * below are default parameters.
 * */
#define DARKLIGHTS_THRESHOLD 30
//#define CALIBRATE_FIT_ERROR_DIF_THRESHOLD 6
//#define HIGHLIGHTS_RADIUS 1
//#define FIT_CANDIDATES 150
//#define HIGHLIGHT_THRESHOLD 240
//#define CenterParam 9

typedef struct CalibParam {
	const char* dataFolder;
	const char* resultFolder;
	const char* calibPrefix;
	const char* dataImgPrefix;
	const char* imgSuffix;

	int numLights;
	int highLightsThreshold;
	int darkLightsThreshold;
	int highLightRaduis;
	int fitCandidateNum;
	int fitErrorThreshold;
	int centerParam;

	CalibParam() {
		dataFolder = NULL;
		resultFolder = NULL;
		numLights = -1;
		fitErrorThreshold = 5;
		highLightRaduis = 1;
		darkLightsThreshold = 30;
		fitCandidateNum = 150;
		highLightsThreshold = 240;
		centerParam = 9;
	}
} CalibParam;

class Calibrator {
public:
	Calibrator(unsigned numLights): numLights(numLights){};
	Calibrator(CalibParam param): param(param){ numLights = param.numLights;};

public:
	float extractHighLight( const Mat &img, float &radius, Vec2f& center, Vec2f& highlight, int lthreshold = DARKLIGHTS_THRESHOLD );
	int loadCalibrationData(const string filename, Vec3f* lightsource);
	int saveCalibrationData(const string filename, const Vec3f* lightsource);
	int calibrateLightsource(const string filename, unsigned numLights);
	int estimateLightPositions(const string filename, unsigned numLights);

	virtual ~Calibrator();

private:

	typedef struct lightInfo
	{
		// L: incident direction vector
		// P: highlight point on the surface
		Vec3f L[2], P[2];
	} lightInfo;

	CalibParam param;
	unsigned numLights;
	vector<Vec2i> hlights, dlights;
	Mat colorImg, watermask;
	int DebugCounter = 1;

	int VcalculateHighligtWithin(const Mat &grayimg, const Mat& highMask, Vec2f & highlight);
	int calculateHighligtWithin(const cv::Mat &grayimg, const cv::Mat& highMask, cv::Vec2f & highlight);
	int calculateIncidentDir(const cv::Vec2f center, float radius, const cv::Vec2f& highlight, Vec3f& dir);
	int calculateHighlightPos(const cv::Vec2f center, float radius, const cv::Vec2f& highlight, Vec3f& dir);
	Vec3f getMedianPoint(const vector<Vec3f> pointsets);
	int estimateLightPosition(lightInfo* temp, Vec3f* lightPositions, int);

	/**
	 * Sort the pixels according to its distance to the center
	 * */
	struct sortByDistance
	{
		Vec2f center;
		sortByDistance(Vec2f center): center(center){};
		bool operator()( const Vec2i &v1, const Vec2i &v2)
		{
			float dx = v1[0] - center[0];
			float dy = v1[1] - center[1];
			float dis1 = dx*dx + dy*dy;

			dx = v2[0] - center[0];
			dy = v2[1] - center[1];
			float dis2 = dx*dx + dy*dy;

			return dis1 < dis2; // ascending sorting
		};

	};

	/**
	 * Vec3i <r, c, luminance>
	 * */
	static inline bool sortByLuminance(const Vec3i &v1, const Vec3i &v2)
	{
		return v1[2] > v2[2]; // descending sorting
	}
};
}
#endif /* SRC_CALIBRATOR_H_ */
