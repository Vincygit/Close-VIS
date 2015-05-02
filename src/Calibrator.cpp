/*
 * Calibrator.cpp
 *
 *  Created on: Mar 13, 2015
 *      Author: vincy
 */

#include "Calibrator.h"

using namespace std;
using namespace cv;

namespace ps {
Calibrator::Calibrator() {
	// TODO Auto-generated constructor stub
	//DebugCounter = 1;
}

Calibrator::~Calibrator() {
	// TODO Auto-generated destructor stub
}

/**
 * update the
 * highlight
 * information.
 *
 * Note that since we use a heuristic method,
 * the highlight information is important and may be updated
 * multiple times in that we need to find the real highlight point.
 *
 * highMask:
 * the highlight should be within this mask.
 * We calculate the most bright points within this mask, and then
 * set the highlight to be the median of all such points.
 * */
int Calibrator::calculateHighligtWithin(const Mat &grayimg, const Mat& highMask, Vec2f & highlight)
{
	cout<< "calculateHighligtWithin.."<<endl;
	Vector<Vec3i> candidates;
	for(int r = 0; r < highMask.rows; r++)
	{
		for(int c = 0; c < highMask.cols; c++)
		{
			// One pass
			if(highMask.at<uchar>(r, c) == 255)
			{
				float sum = 0;
				for(int i = r - HIGHLIGHTS_RADIUS/2; i <= r + HIGHLIGHTS_RADIUS/2; i++)
				{
					for(int j = c - HIGHLIGHTS_RADIUS/2; j <= c + HIGHLIGHTS_RADIUS/2; j++)
					{
						sum += grayimg.at<uchar>(i, j);
					}
				}
				candidates.push_back(Vec3i(r,c,sum/(HIGHLIGHTS_RADIUS*HIGHLIGHTS_RADIUS)));
			}
		}
	}
	// sort all the candidate points.
	sort(candidates.begin(), candidates.end(), ps::Calibrator::sortByLuminance);

	// now we pick real candidates.
	// TODO: maybe we could calculate the gradients and pick up the point
	// that has the biggest gradient and use its value as the threshold.
	// but now i just do the easy job.
	uchar brightThreshold = 0u;
#ifdef USE_RATIO
	brightThreshold = candidates[HIGHLIGHT_RATIO * candidates.size()][2];
#else
	if(candidates.size() <= 2 * HIGHLIGHT_COUNT)
	{
		brightThreshold = candidates[HIGHLIGHT_RATIO * candidates.size()][2];
		cout << " Too few candidates, forced to  pick up ratio points" << endl;
	} else {
		// pick up the last point that is with such luminance
		brightThreshold = candidates[HIGHLIGHT_COUNT][2];
	}
#endif
	cout<< "calculating.."<<endl;
	// second pass, collect all points that are bright enough as the real candidates
	int Xcount = 0, Ycount = 0;
	unsigned i;
	Mat debugMask(grayimg.size(), CV_8U, Scalar(0));

	for(i = 0; candidates[i][2] >= brightThreshold && i < candidates.size(); i++)
	{
		debugMask.at<uchar>(candidates[i][0], candidates[i][1]) = 255;
		Xcount += candidates[i][0];
		Ycount += candidates[i][1];
	}

	//TODO: calculate the median instead of mean.

	if(i == 0 ) {
		cout<< "*Error: no candidates detected.."<<endl;
	}

	highlight[0] = (float)Xcount / i;
	highlight[1] = (float)Ycount / i;

	char buffer[100];
	sprintf(buffer, RESULT_FOLDER"debug_high_mask_%d.jpg", DebugCounter);
	imwrite(buffer, debugMask);
	return 1;
}

/**
 * Optimized version,
 * use median filter instead of mean
 * */
int Calibrator::VcalculateHighligtWithin(const Mat &grayimg, const Mat& highMask, Vec2f & highlight)
{
	cout<< "V->calculateHighligtWithin.."<<endl;
	Vector<Vec3i> candidates;
	vector<int> xIdx, yIdx;
	//	cout<<"rows:"<<highMask.rows<<endl;
	for(int r = 0; r < highMask.rows; r++)
	{
		for(int c = 0; c < highMask.cols; c++)
		{
			// One pass
			if(highMask.at<uchar>(r, c) == 255)
			{
				//				float sum = 0;
				//				for(int i = r - HIGHLIGHTS_RADIUS/2; i <= r + HIGHLIGHTS_RADIUS/2; i++)
				//				{
				//					for(int j = c - HIGHLIGHTS_RADIUS/2; j <= c + HIGHLIGHTS_RADIUS/2; j++)
				//					{
				//						sum += grayimg.at<uchar>(i, j);
				//					}
				//				}
				if(grayimg.at<uchar>(r, c) > HIGHLIGHT_THRESHOLD)
				{
					xIdx.push_back(r);
					yIdx.push_back(c);
					candidates.push_back(Vec3i(r, c, grayimg.at<uchar>(r, c)));
				}
			}
		}
	}

	if(candidates.size() <= 1)
	{
		cout<<"ERRORROORR-> Please decrease the highlight threshold"<<endl;
		return 0;
	}

	// sort all the candidate points.
	sort(candidates.begin(), candidates.end(), ps::Calibrator::sortByLuminance);
	sort(xIdx.begin(),xIdx.end());
	sort(yIdx.begin(),yIdx.end());

	// now we pick real candidates.
	float length = xIdx.size()/2.0f;
	cout << length << endl;
	highlight[0] = (float)(xIdx[ceil(length)] + xIdx[floor(length)])/2.0f;
	highlight[1] = (float)(yIdx[ceil(length)] + yIdx[floor(length)])/2.0f;

	cout<< "calculated median of hightlight point sets."<<highlight<<endl;
	// second pass, collect all points that are bright enough as the real candidates


	return 1;
}

/**
 * fit a circle and use its center as the highlight point
 * we first use pick up all the points that are with high chrome
 * img: filtered gray image
 * @return: fitting error
 * */
float Calibrator::extractHighLight( const Mat &img, float &radius, Vec2f& center, Vec2f& highlight, int lthreshold )
{
	float fittingError = .0f;
	int height = img.rows, width = img.cols;

	// since we have very robust scenario that the dark point would be on the ball
	// we can pick them up directly.
	// TODO:: use a color ball to achieve the same thing.
	dlights.clear();
	for( int h = height/CenterParam; h < height*(CenterParam-1)/CenterParam; h++)
	{
		for( int w = width/CenterParam; w < width*(CenterParam-1)/CenterParam; w++)
		{
//			if( img.at<uchar>(h, w) < lthreshold)
//			{
//				dlights.push_back( Vec2i(h, w));
//			}
			int sum = 0;
			for(int i = h - HIGHLIGHTS_RADIUS/2; i <= h + HIGHLIGHTS_RADIUS/2; i++)
			{
				for(int j = w - HIGHLIGHTS_RADIUS/2; j <= w + HIGHLIGHTS_RADIUS/2; j++)
				{
					sum += img.at<uchar>(i, j);
				}
			}
			if( sum < lthreshold * (HIGHLIGHTS_RADIUS*HIGHLIGHTS_RADIUS))
			{
				dlights.push_back( Vec2i(h, w));
			}
		}
	}

	if(dlights.size() == 0) {
		cout<< "No darkLights detected, please reset the threshold value." << endl;
		return 0;
	}

	Mat markers(img.size(), CV_8U, cv::Scalar(0));
	for(unsigned k = 0; k < dlights.size(); k++)
	{
		markers.at<uchar>(dlights[k][0],dlights[k][1]) = 128;
	}

	char resbuf[100];
	sprintf(resbuf,RESULT_FOLDER"darklights%d.jpg", DebugCounter);
	imwrite(resbuf, markers);

	// find the contour by using marker
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours( markers, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	cout<< "Contours found." << endl;
	/// Choose the bigest contour
	unsigned idx = 0, length = 0;
	for(unsigned i = 0; i < contours.size(); i++)
	{
		if(contours[i].size() > length)
		{
			idx = i;
			length = contours[i].size();
		}
	}

	/**
	 * TODO: can also use convex. But may introduce more problems.
		std::vector<Point> hull;
		convexHull(Mat(contours[3]),hull);
		std::vector<cv::Point>::const_iterator it= hull.begin();
		while(it != (hull.end()-1))
		{
			line(result,*it,*(it+1),Scalar(0),2);
			++it;
		}
		line(result,*(hull.begin()),*(hull.end()-1),Scalar(0),2);
	 */
	Mat drawing = Mat::zeros( markers.size(), CV_32S );
	drawing = Scalar::all(0);
	/// add outline marker
	drawContours( drawing, contours, (int)idx, Scalar(1), -1, 8, hierarchy, INT_MAX);
	cout<< "Running watershed segmentation..." <<endl;

#ifndef USEWATERSHED
	cv::rectangle(drawing,cv::Point(drawing.cols/CenterParam,drawing.rows/CenterParam), cv::Point(drawing.cols/CenterParam*(CenterParam-1), drawing.rows/CenterParam*(CenterParam-1)),cv::Scalar(2),FittedCircleThickness);
	// use watershed
	watershed(colorImg, drawing);

	Mat wshed(drawing.size(), CV_8U);

	// paint the watershed image
	for( int i = 0; i < drawing.rows; i++ )
		for( int j = 0; j < drawing.cols; j++ )
		{
			int index = drawing.at<int>(i,j);
			if( index == -1 )
				wshed.at<uchar>(i,j) = 0;
			else if( index <= 0 || index > 1 )
				wshed.at<uchar>(i,j) = 128;
			else
				wshed.at<uchar>(i,j) = 255;
		}

	sprintf(resbuf,RESULT_FOLDER"watershed%d.jpg", DebugCounter);
	imwrite(resbuf, wshed);
#endif

	// after the first segmentation, we calculate our bright center.
	VcalculateHighligtWithin(img, wshed, highlight);

	// mark the rough highlight point.
//	cv::rectangle(wshed,cv::Point(highlight[1]-5,highlight[0]-5), cv::Point(highlight[1]+5,highlight[0]+5),cv::Scalar(20),-1);
//	imwrite(RESULT_FOLDER"watershed_with_highlight.jpg", wshed);

	// pickup the candidate points used to fit the circle
	vector<Vec2i> candidates;
	for( int i=wshed.rows/CenterParam;i<wshed.rows*(CenterParam-1)/CenterParam;i++)
	{
		for(int j=wshed.cols/CenterParam;j<wshed.cols*(CenterParam-1)/CenterParam;j++)
		{
			if(wshed.at<uchar>(i, j) == 0)
			{
				candidates.push_back(Vec2i(i,j));
			}
		}
	}

	// TODO: eliminate the use of global variable
	std::sort(candidates.begin(), candidates.end(), ps::Calibrator::sortByDistance(highlight));
	Mat circle_margin = Mat::zeros( markers.size(), CV_8U );
	for(unsigned i = 0; i < FIT_CANDIDATES && i < candidates.size(); i++)
	{
		circle_margin.at<uchar>(candidates[i][0], candidates[i][1]) = 255;
	}

	sprintf(resbuf,RESULT_FOLDER"circle_margin%d.jpg", DebugCounter);
	imwrite(resbuf, circle_margin);

	if(candidates.size() > FIT_CANDIDATES)
		candidates.resize(FIT_CANDIDATES);

#ifdef HOUGHTRANS
	// use hough transform
	vector<Vec3f> circles;
	HoughCircles(img, circles, CV_HOUGH_GRADIENT, 1.15, 100, 100, 60, 100, 300);
	drawing.convertTo(drawing, CV_8UC3);
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Vec3i c = circles[i];
		cout<< c[2] << "radius "<<endl;
		circle( colorImg, Point(c[0], c[1]), c[2], Scalar(0,0,255), FittedCircleThickness, CV_AA);
		circle( colorImg, Point(c[0], c[1]), 2, Scalar(0,255,0), FittedCircleThickness, CV_AA);
	}

	imwrite(RESULT_FOLDER"houghCircles.jpg", colorImg);
#elif defined MSE
	// use minimum squre error to fit the circle.
	// the benefit is that we will only have one single solution.
	double points[2 * candidates.size()];
	for(unsigned i = 0; i < candidates.size(); i++)
	{
		points[2*i] = candidates[i][0];
		points[2*i+1] = candidates[i][1];
	}

	BestFitIO input;
	input.numPoints = candidates.size();
	input.points = points;

	// Output settings in this simple example left with default values.
	BestFitIO output;

	// Create the best-fitting circle object, and make it do the work.
	int type = BestFitFactory::Circle;
	ostringstream fitVerb;
	BestFit *b = BestFitFactory::Create(type, fitVerb);
	b->Compute(input, output);

	// get the output stream and parse them.
	std::string s = fitVerb.str();
	//	cout << s << endl;
	vector<string> lines = split(s,'\n');
	vector<string> varline = split(lines[6], ' ');
	fittingError = atof(varline[varline.size() - 1].c_str());
	cout << "fit error:" << fittingError << endl;
	// get the properties about the circle
	center[0] = output.outputFields[BestFitIO::CircleCentreX];
	center[1] = output.outputFields[BestFitIO::CircleCentreY];
	radius = output.outputFields[BestFitIO::CircleRadius];

	cout<<"xCol= "<<center[0] <<" yRow= "<<center[1]<<" radius= " << radius<< endl;

	delete b;
#endif

	// re-calculate the highlight point.
	Mat highMask(img.size(), CV_8U, Scalar(0));
	circle( highMask, Point(center[1], center[0]), (int)radius + 1, Scalar(255), -1, CV_AA);
	VcalculateHighligtWithin(img, highMask, highlight);

	circle( colorImg, Point(highlight[1], highlight[0]), 3, Scalar(255,255,0), FittedCircleThickness, CV_AA);
	circle( colorImg, Point(center[1], center[0]), (int)radius, Scalar(0,0,255), FittedCircleThickness, CV_AA);
	circle( colorImg, Point(center[1], center[0]), 3, Scalar(0,255,0), FittedCircleThickness, CV_AA);
	circle( colorImg, Point(highlight[1], highlight[0]), 3, Scalar(255,0,0), FittedCircleThickness, CV_AA);

	sprintf(resbuf,RESULT_FOLDER"result%d.jpg", DebugCounter++);
	imwrite(resbuf, colorImg);

	cout<< "Circle fitted, " << endl;

	return fittingError;
}


/**
 * method1:
 * use watershed segmentation method.
 *
 * method2:
 * use meanshift segmentation, during which we take the color information into consideration.
 *
 * method3:
 * using GMM to model the background thus extract the foreground
 * then we use heuristic way to pick some points for the sphere fitting.
 * it may be better if we use all four input images then calculate the result.
 * Because we can better model the background.
 *
 * method4:
 * we can first take pictures in very bright environment,
 * this will make the GMM method even more useful.
 *
 * The output of all the three method are to output an edge, which can then
 * be used to calculate the parameters about the sphere.
 * : center point and the radius.
 *
 * which is better ??
 * */
int Calibrator::calculateIncidentDir(const cv::Vec2f center, float radius, const cv::Vec2f& highlight, Vec3f& dir)
{
	cout<< "calculate incident dir from center:"<< center << "with highlight: " << highlight<<endl;
	// note the coordinate here is for Matrix, not image.
	float x = (highlight[0] - center[0]) / radius;
	float y = (highlight[1] - center[1]) / radius;
	float z = sqrt(1 - x*x - y*y);

	dir[0] = 2 * z * x;
	dir[1] = 2 * z * y;
	dir[2] = 2 * z * z - 1;

	return radius > 0;
}

int Calibrator::saveCalibrationData(const string filename, const Vec3f* lightsource)
{
	ofstream of(filename.c_str());

	for(unsigned i = 0; i < numImages; i++)
	{
		of << lightsource[i][0] <<" "<<lightsource[i][1] <<" "<<lightsource[i][2] <<" " << "\n";
		cout << lightsource[i][0] <<" "<<lightsource[i][1] <<" "<<lightsource[i][2] <<" " << "\n";
	}
	of.close();
	return 1;
}

int Calibrator::loadCalibrationData(const string filename, Vec3f* lightsource)
{
	ifstream is(filename.c_str());
	//	char buf[100];
	//	while(is.getline(buf, 100)) {
	//		cout <<buf<<endl;
	//	}

	for(unsigned i = 0; i < numImages; i++)
	{
		is >> lightsource[i][0] >> lightsource[i][1]>> lightsource[i][2];
		cout << "Loading Data:" << endl;
		cout << lightsource[i] << endl;
	}
	is.close();
	return 1;
}

int Calibrator::calibrateLightsource(const string filename, unsigned numImages)
{
	/**
	 * Version 1.0:
	 * I will use all four images to do this calibration.
	 * The circle will be the one that with the smallest fitting error.
	 * */
	float ferrors[numImages], minError = FLT_MAX; // note that the information stored in here is in matrix format
	Vec3f lightsource[numImages], ball;
	Mat filteredimg[numImages];
	unsigned f;
	char buffer[100];
	for( f = 0; f < numImages; f++)
	{
		sprintf(buffer, DATA_FOLDER"IMG_%d.JPG", (f+1));
		// load gray image is enough
		colorImg = imread(buffer);
		if( colorImg.data == NULL) {
			cout<< "Failed to load image." <<endl;
		}
		Mat grayImg, filImag;
		cvtColor(colorImg, grayImg,CV_BGR2GRAY);
		imwrite( RESULT_FOLDER"gray.jpg", grayImg );

		//		cout<<colorImg.type()<<endl;

		// Step 1: preprocessing
		//FIXME
		//		filterImage( grayImg, filImag);
		filImag = grayImg;
		grayImg.convertTo(filteredimg[f], CV_32F);
		imwrite( RESULT_FOLDER"blur.jpg", filImag );

		// Step 2: locate the ball
		Vec2f center, highlight;
		float radius;

		ferrors[f] = extractHighLight(filImag, radius, center, highlight);

		if(minError > ferrors[f])
		{
			minError = ferrors[f];
			ball[0] = center[0];

			// FIXME::
			// FIXME::
			// FIXME::
			// FIXME::
			// FIXME::
			ball[1] = center[1];
			ball[2] = radius;
		}

		cout<< " highlight point is: " << highlight <<endl <<endl;
		// Step 3: calculate the incident direction
		// TODO: refine the light source properties

		//FIXME exchange coordinates
		float tp = highlight[1];
		highlight[1] = highlight[0];
		highlight[0] = tp;

		tp = center[1];
		center[1] = center[0];
		center[0] = tp;
		calculateIncidentDir(center, radius, highlight, lightsource[f]);
	}

	cout<<"****Calibrated Real Center is:"<<ball<<endl;
	for( f = 0; f < numImages; f++)
	{
		// second pass. filter out the wrong cases
		if(abs(minError - ferrors[f]) > CALIBRATE_FIT_ERROR_DIF_THRESHOLD)
		{
			// refine the highlight point
			Vec2f highlight;
			Mat highMask(filteredimg[f].size(), CV_8U, Scalar(0));
			circle( highMask, Point(filteredimg[f].rows - 1 - ball[1], ball[0]), (int)ball[2] + 1, Scalar(255), -1, CV_AA);
			VcalculateHighligtWithin(filteredimg[f], highMask, highlight);

			//FIXME exchange coordinates
			float tp = highlight[1];
			highlight[1] = highlight[0];
			highlight[0] = tp;

			calculateIncidentDir(Vec2f(ball[1], ball[0]), ball[2], highlight, lightsource[f]);

			circle( filteredimg[f], Point(highlight[1], highlight[0]), 3, Scalar(255,255,0), FittedCircleThickness, CV_AA);
			circle( filteredimg[f], Point(filteredimg[f].rows - 1 - ball[1], ball[0]), (int)ball[2], Scalar(0,0,255), FittedCircleThickness, CV_AA);
			circle( filteredimg[f], Point(filteredimg[f].rows - 1 - ball[1], ball[0]), 3, Scalar(0,255,0), FittedCircleThickness, CV_AA);
			circle( filteredimg[f], Point(highlight[1], highlight[0]), 3, Scalar(255,0,0), FittedCircleThickness, CV_AA);

			char resbuf[100];
			sprintf(resbuf,RESULT_FOLDER"result%d.jpg", f + 1);
			imwrite(resbuf, filteredimg[f]);
		}
	}

	saveCalibrationData(filename, lightsource);

	return 1;
}

}
