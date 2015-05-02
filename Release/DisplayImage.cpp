
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include "BestFit.h"

using namespace std;
using namespace cv;
#define HIGHLIGHTS_RADIUS 3
#define DARKLIGHTS_THRESHOLD 30
#define USE_HIGHLIGHT 0
#define FIT_CANDIDATES 500
#define MSE
#define USE_WATERSHED
#define HIGHLIGHT_RATIO 0.1
#define HIGHLIGHT_COUNT 200
#define NUM_IMAGES 4

#define CALIBRATE_FIT_ERROR_DIF_THRESHOLD 100

#define RESULT_FOLDER "/home/vincy/research/data/result/"
#define DATA_FOLDER "/home/vincy/research/data/"

vector<Vec2i> hlights, dlights;
Mat colorImg, watermask;
int DebugCounter = 0;

/**
 * Helper functions
 */
static inline void help(char* progName)
{
	cout << endl
			<<  "This is an demo for auto-calibration program of photometric stereo. " << endl
			<< progName << " [image_name] "                       << endl << endl;
}

/**
 * TODO:
 * How to eliminate this global variable?
 * */
static Vec2f sortHighLight;

static inline bool sortByDistance( const Vec2i &v1, const Vec2i &v2)
{
	float dx = v1[0] - sortHighLight[0];
	float dy = v1[1] - sortHighLight[1];
	float dis1 = dx*dx + dy*dy;

	dx = v2[0] - sortHighLight[0];
	dy = v2[1] - sortHighLight[1];
	float dis2 = dx*dx + dy*dy;

	return dis1 < dis2; // ascending sorting
}


std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

/**
 * Vec3i <r, c, luminance>
 * */
bool sortByLuminance(const Vec3i &v1, const Vec3i &v2)
{
	return v1[2] > v2[2]; // descending sorting
}

/**
 * preprocess the input image.
 * Low pass filter. de-noise.
 * */
int filterImage(Mat& img, Mat& output, int windowSize=3, float sigma=10)
{
	cv::GaussianBlur(img,output,cv::Size(windowSize,windowSize),sigma);
	return 1;
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
int calculateHighligtWithin(const Mat &grayimg, Mat& highMask, Vec2f & highlight)
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
	sort(candidates.begin(), candidates.end(), sortByLuminance);

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
	sprintf(buffer, RESULT_FOLDER"%d.jpg", DebugCounter++);
	imwrite(buffer, debugMask);
	return 1;
}

/**
 * fit a circle and use its center as the highlight point
 * we first use pick up all the points that are with high chrome
 * img: filtered gray image
 * @return: fitting error
 * */
float extractHighLight( const Mat &img, float &radius, Vec2f& center, Vec2f& highlight, int lthreshold = DARKLIGHTS_THRESHOLD )
{
	float fittingError = .0f;
	int height = img.rows, width = img.cols;

	// since we have very robust scenario that the dark point would be on the ball
	// we can pick them up directly.
	// TODO:: use a color ball to achieve the same thing.
	for( int h = height/4; h < height/4*3; h++)
	{
		for( int w = width/4; w < width/4*3; w++)
		{
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
	imwrite(RESULT_FOLDER"highlights.jpg", markers);

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
	cv::rectangle(drawing,cv::Point(drawing.cols/4,drawing.rows/4), cv::Point(drawing.cols/4*3, drawing.rows/4*3),cv::Scalar(2),10);
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
	imwrite(RESULT_FOLDER"watershed.jpg", wshed);
#endif

	// after the first segmentation, we calculate our bright center.
	calculateHighligtWithin(img, wshed, highlight);

	// mark the rough highlight point.
	cv::rectangle(wshed,cv::Point(highlight[1]-5,highlight[0]-5), cv::Point(highlight[1]+5,highlight[0]+5),cv::Scalar(20),-1);
	imwrite(RESULT_FOLDER"watershed_with_highlight.jpg", wshed);

	// pickup the candidate points used to fit the circle
	vector<Vec2i> candidates;
	for( int i=wshed.rows/4;i<wshed.rows*3/4;i++)
	{
		for(int j=wshed.cols/4;j<wshed.cols*3/4;j++)
		{
			if(wshed.at<uchar>(i, j) == 0)
			{
				candidates.push_back(Vec2i(i,j));
			}
		}
	}

	// TODO: eliminate the use of global variable
	sortHighLight = highlight;
	std::sort(candidates.begin(), candidates.end(), sortByDistance);
	Mat circle_margin = Mat::zeros( markers.size(), CV_8U );
	for(unsigned i = 0; i < FIT_CANDIDATES && i < candidates.size(); i++)
	{
		circle_margin.at<uchar>(candidates[i][0], candidates[i][1]) = 255;
	}
	imwrite(RESULT_FOLDER"circle_margin.jpg", circle_margin);

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
		circle( colorImg, Point(c[0], c[1]), c[2], Scalar(0,0,255), 10, CV_AA);
		circle( colorImg, Point(c[0], c[1]), 2, Scalar(0,255,0), 10, CV_AA);
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
	cout << s << endl;
	vector<string> lines = split(s,'\n');
	vector<string> varline = split(lines[6], ' ');
	fittingError = atof(varline[varline.size() - 1].c_str());
	cout << "fit error:" << fittingError << endl;
	// get the properties about the circle
	center[0] = output.outputFields[BestFitIO::CircleCentreX];
	center[1] = output.outputFields[BestFitIO::CircleCentreY];
	radius = output.outputFields[BestFitIO::CircleRadius];

	cout<<"x= "<<center[1] <<" y= "<<center[0]<<" radius= " << radius<< endl;

	delete b;
#endif

	// re-calculate the highlight point.
	Mat highMask(img.size(), CV_8U, Scalar(0));
	circle( highMask, Point(center[1], center[0]), (int)radius + 1, Scalar(255), -1, CV_AA);
	calculateHighligtWithin(img, highMask, highlight);

	circle( colorImg, Point(highlight[1], highlight[0]), 3, Scalar(255,255,0), 10, CV_AA);
	circle( colorImg, Point(center[1], center[0]), (int)radius, Scalar(0,0,255), 10, CV_AA);
	circle( colorImg, Point(center[1], center[0]), 3, Scalar(0,255,0), 10, CV_AA);
	circle( colorImg, Point(highlight[1], highlight[0]), 3, Scalar(255,0,0), 10, CV_AA);

	imwrite(RESULT_FOLDER"result.jpg", colorImg);

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
int calculateIncidentDir(const cv::Vec2f center, float radius, const cv::Vec2f& highlight, Vec3f& dir)
{
	// note the coordinate here is for Matrix, not image.
	float x = (highlight[0] - center[0]) / radius;
	float y = (highlight[1] - center[1]) / radius;
	float z = sqrt(1 - x*x - y*y);

	dir[0] = 2 * z * x;
	dir[1] = 2 * z * y;
	dir[2] = 2 * z * z - 1;

	return radius > 0;
}

/***
 * process the input image.
 * 1. normalize image intensities, get the shadowMask
 * 2. deal with the black pixels.
 * */
int processInputImages(cv::Mat* shadowMask, const cv::Mat* srcImages, float shadowThreshold = 0.1)
{
	float MaxVal = 255.0f;
	// TODO: apply the percentile method
	//	for(unsigned f = 0; f < NUM_IMAGES; f++)
	//	{
	//		// first pass, find the maximum value
	//		//
	//	}
	float realThresh = MaxVal * shadowThreshold;

	int erosion_type = MORPH_ELLIPSE, erosion_size = 5;

	Mat element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ), Point( erosion_size, erosion_size ) );

	// second pass, uniform all pixels
	for(unsigned f = 0; f< NUM_IMAGES; f++)
	{
		if(shadowMask[f].data == NULL) {
			shadowMask[f].create(srcImages[f].rows, srcImages[f].cols, CV_8U);
		}

		for( int h = 0; h < srcImages[f].rows; h++)
		{
			for( int w = 0; w < srcImages[f].cols; w++)
			{
				if(srcImages[f].at<float>(h, w) <= realThresh)
				{
					shadowMask[f].at<uchar>(h, w) = 0;
				} else {
					shadowMask[f].at<uchar>(h, w) = 1;
				}
			}
		}

		// Now erode the shadow masks to handle de-mosaiking artifact near shadow boundary.
		/// Apply the erosion operation
		//		cv::erode( shadowMask[f], shadowMask[f], element );
	}

	return 1;
}

/**
 * apply photometric stereo
 * Just use linear least square error.
 * */
int applyPhotometricStereo(const cv::Mat* shadowMask, const cv::Mat* srcImages, const cv::Vec3f* srcLight, cv::Mat& albedo, cv::Mat& norm)
{
	// first use the shadow mask to filter the srcImage pixels.
	// we only consider pixels that appear more than 2 times >=3
	// the result is merged into a new matrix
	int matRows = srcImages[0].rows, matCols = srcImages[0].cols;

	Mat srcLightMat(NUM_IMAGES, 3, CV_32F); // 3xM
	for(int f = 0; f < NUM_IMAGES; f++)
	{
		srcLightMat.at<float>(f,0) = srcLight[f][0];
		srcLightMat.at<float>(f,1) = srcLight[f][1];
		srcLightMat.at<float>(f,2) = srcLight[f][2];
	}

	unsigned totalPoints = matRows * matCols;
	// now apply the mask into the intensity map
	Mat reflectMat;

	// process the input mask
	Mat totalMask(shadowMask[0].size(), CV_8U, Scalar(0));
	Mat intensityMat(NUM_IMAGES, totalPoints, CV_32F, Scalar(0));

	for(int f = 0; f < NUM_IMAGES; f++)
	{
		totalMask += shadowMask[f] * (1<<f);
	}

	for(int f = 0; f < NUM_IMAGES; f++)
	{
		for(int h = 0; h < matRows; h++)
		{
			for(int w = 0; w < matCols; w++)
			{
				//				if(totalMask.at<uchar>(h, w) == ((1<<NUM_IMAGES) -1 ))
				intensityMat.at<float>(f, h*matCols + w) = srcImages[f].at<float>(matRows - 1 - h, w) / 255.0;
				//				else
				//					intensityMat.at<float>(f, h*matCols + w) = NAN;
			}
		}
	}

	// TODO: consider all cases when the pixel appear at least 3 times
	// FIXME: currently i only consider points that appear in all images
	Mat svd = srcLightMat.inv(DECOMP_SVD);
	cout<<svd<<endl;
	reflectMat = svd*(intensityMat);

	//	ofstream ofer(DATA_FOLDER"intense.txt");
	//	for(int f = 0; f < NUM_IMAGES; f++)
	//	{
	//		for(int w = 0; w < matCols; w++)
	//		{
	//			for(int h = 0; h < matRows; h++)
	//			{
	//				ofer<<intensityMat.at<float>(f, h*matCols + w)<<" ";
	//			}
	//		}
	//		ofer << "\n";
	//	}
	//	ofer.close();

	if(albedo.data == NULL)
	{
		albedo.create(matRows, matCols, CV_32F);
		for(int h = 0; h < matRows; h++)
		{
			for(int w = 0; w < matCols; w++)
			{
				int refIdx = h*matCols + w;
				float x = reflectMat.at<float>(0, refIdx);
				float y = reflectMat.at<float>(1, refIdx);
				float z = reflectMat.at<float>(2, refIdx);
				albedo.at<float>(h, w) = sqrt(x*x + y*y + z*z);
			}
		}
	}

	if(norm.data == NULL)
	{
		norm.create(matRows, matCols, CV_32FC3);
		for(int h = 0; h < matRows; h++)
		{
			for(int w = 0; w < matCols; w++)
			{
				int refIdx = h*matCols + w;

				for( int i = 0; i < 3; i++)
				{
					float x = reflectMat.at<float>(i, refIdx);
					norm.at<Vec3f>(h, w)[i] = x / albedo.at<float>(h, w);
				}
			}
		}
	}

	return 1;
}

void integrate( float *P,  float *Q,  float *Z, int width, int height, float mu = 0.4) {

	/* get current i,j position in image */
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			if ((i != 0) || (j != 0)) {
				float u = sin((float)(i*2*M_PI/height));
				float v = sin((float)(j*2*M_PI/width));
				float uv = pow(u,2)+pow(v,2);
				float d = uv + mu*(pow(u,4)+pow(v,4));
				/* offset = (row * numCols * numChannels) + (col * numChannels) + channel */
				Z[(i*width*2)+(j*2)+(0)] = ((u+mu*pow(u,3))*P[(i*width*2)+(j*2)+(1)]  + (v+mu*pow(v,3))*Q[(i*width*2)+(j*2)+(1)]) / d;
				Z[(i*width*2)+(j*2)+(1)] = (-(u+mu*pow(u,3))*P[(i*width*2)+(j*2)+(0)] - (v+mu*pow(v,3))*Q[(i*width*2)+(j*2)+(0)]) / d;
				//				float d = uv + mu*pow(uv,2);
				//				/* offset = (row * numCols * numChannels) + (col * numChannels) + channel */
				//				Z[(i*width*2)+(j*2)+(0)] = (u*P[(i*width*2)+(j*2)+(1)]  + v*Q[(i*width*2)+(j*2)+(1)]) / d;
				//				Z[(i*width*2)+(j*2)+(1)] = (-u*P[(i*width*2)+(j*2)+(0)] - v*Q[(i*width*2)+(j*2)+(0)]) / d;
			}
		}
	}
}

void Vintegrate( Mat &P,  Mat &Q,  Mat&Z, int width, int height, float mu = 0.4) {

	/* get current i,j position in image */
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			if ((i != 0) || (j != 0)) {
				float u = sin((float)(i*2*M_PI/height));
				float v = sin((float)(j*2*M_PI/width));
				float uv = pow(u,2)+pow(v,2);
				float d = uv + mu*(pow(u,4)+pow(v,4));
				/* offset = (row * numCols * numChannels) + (col * numChannels) + channel */
				Z.at<Vec2f>(i, j)[0] = ((u+mu*pow(u,3))*P.at<Vec2f>(i, j)[1]  + (v+mu*pow(v,3))*Q.at<Vec2f>(i, j)[1]) / d;
				Z.at<Vec2f>(i, j)[1] = (-(u+mu*pow(u,3))*P.at<Vec2f>(i, j)[0] - (v+mu*pow(v,3))*Q.at<Vec2f>(i, j)[0]) / d;
			}
		}
	}
}

cv::Mat getGlobalHeights(cv::Mat Pgrads, cv::Mat Qgrads) {

	cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));

	cv::dft(Pgrads, P, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(Qgrads, Q, cv::DFT_COMPLEX_OUTPUT);

	cout<<Pgrads.at<float>(0,0)<<"heysss"<<endl;
	cout<<((float *)(P.data))[0]<<endl;
	//	integrate((float *)P.data, (float *)Q.data, (float *)Z.data,Pgrads.cols, Pgrads.rows);
	Vintegrate(P, Q, Z,Pgrads.cols, Pgrads.rows);
	//	/* creating OpenCL buffers */
	//	size_t imgSize = sizeof(float) * (height*width*2); /* 2 channel matrix */
	//	cl_P = cl::Buffer(context, CL_MEM_READ_ONLY, imgSize, NULL, &error);
	//	cl_Q = cl::Buffer(context, CL_MEM_READ_ONLY, imgSize, NULL, &error);
	//	cl_Z = cl::Buffer(context, CL_MEM_WRITE_ONLY, imgSize, NULL, &error);
	//
	//	/* pushing data to CPU */
	//	queue.enqueueWriteBuffer(cl_P, CL_TRUE, 0, imgSize, P.data, NULL, &event);
	//	queue.enqueueWriteBuffer(cl_Q, CL_TRUE, 0, imgSize, Q.data, NULL, &event);
	//	queue.enqueueWriteBuffer(cl_Z, CL_TRUE, 0, imgSize, Z.data, NULL, &event);
	//
	//	/* set kernel arguments */
	//	integKernel.setArg(0, cl_P);
	//	integKernel.setArg(1, cl_Q);
	//	integKernel.setArg(2, cl_Z);
	//	integKernel.setArg(3, width);
	//	integKernel.setArg(4, height);
	//	integKernel.setArg(5, lambda);
	//	integKernel.setArg(6, mu);
	//	/* wait for command queue to finish before continuing */
	//	queue.finish();
	//
	//	/* executing kernel */
	//	queue.enqueueNDRangeKernel(integKernel, cl::NullRange, cl::NDRange(height, width), cl::NullRange, NULL, &event);
	//
	//	/* reading back from CPU */
	//	queue.enqueueReadBuffer(cl_Z, CL_TRUE, 0, imgSize, Z.data);

	/* setting unknown average height to zero */
	Z.at<cv::Vec2f>(0, 0)[0] = 0.0f;
	Z.at<cv::Vec2f>(0, 0)[1] = 0.0f;

	cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE |  cv::DFT_REAL_OUTPUT);

	return Z;
}


// Now Port the matlab codes directly
cv::Mat performFCAlgo(const cv::Mat&Pmat, const cv::Mat&Qmat)
{
	int m = Pmat.rows << 1;
	int n = Pmat.cols << 1;

	// Perform copy-flip for non-periodic depth.
	cv::Mat cfPmat(m, n, CV_32F), cfQmat(m, n, CV_32F);
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			if(i < (m>>1) && j < (n>>1) )
			{
				// second quadrant
				cfPmat.at<float>(i, j) = Pmat.at<float>(i, j);
				cfQmat.at<float>(i, j) = Qmat.at<float>(i, j);
			} else if(i >= (m>>1) && j < (n>>1) ) {
				// third quadrant
				cfPmat.at<float>(i, j) = Pmat.at<float>(m-1-i, j);
				cfQmat.at<float>(i, j) = -Qmat.at<float>(m-1-i, j);
			} else if(i >= (m>>1) && j >= (n>>1) ) {
				// fourth quadrant
				cfPmat.at<float>(i, j) = -Pmat.at<float>(m-1-i, n-1-j);
				cfQmat.at<float>(i, j) = -Qmat.at<float>(m-1-i, n-1-j);
			} else {
				// first quadrant
				cfPmat.at<float>(i, j) = -Pmat.at<float>(i, n-1-j);
				cfQmat.at<float>(i, j) = Qmat.at<float>(i, n-1-j);
			}
		}
	}
	// generate frequency indices
	cv::Mat uMat, vMat;
	cv::Mat du(1, n, CV_32F), dv( 1, n, CV_32F,Scalar(-1));
	const Mat oneLine( 1, n, CV_32F,Scalar(1));

	float * data = du.ptr<float>(0);
	for(int i = 0; i < n; i++)
	{
		if(i < (n>>1)) // notice cannot equal here;
			data[i] = i;
		else
			data[i] = i-n;
	}

	for(int i = 0; i < m; i++)
	{
		uMat.push_back(du);

		dv+=oneLine;
		if( i == (m>>1))
			dv = cv::Mat::ones(1,n,CV_32F) * (-(m>>1));
		vMat.push_back(dv);
	}

	// now extend the complex part
	cv::Mat P(m, n, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Q(m, n, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Z(m, n, CV_32FC2, cv::Scalar::all(0));

	cv::dft(cfPmat, P, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(cfQmat, Q, cv::DFT_COMPLEX_OUTPUT);


	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			if( i + j > 0)
			{
				float u = uMat.at<float>(i,j);
				float v = vMat.at<float>(i,j);
				double d = 2*M_PI*( pow( (u / n), 2 ) + pow( (v / m), 2 ) );

				Z.at<Vec2f>(i, j)[0] = (u*P.at<Vec2f>(i,j)[1]/n + v*Q.at<Vec2f>(i,j)[1]/m) / d;
				Z.at<Vec2f>(i, j)[1] = -(u*P.at<Vec2f>(i,j)[0]/n + v*Q.at<Vec2f>(i,j)[0]/m) / d;
			}
		}
	}

	// set DC component
	Z.at<cv::Vec2f>(0, 0)[0] = 0.0f;
	Z.at<cv::Vec2f>(0, 0)[1] = 0.0f;

	// reverse the transform
	cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE |  cv::DFT_REAL_OUTPUT);

	// recover the non-periodic depth
	cv::Mat depthMap(m>>1, n>>1, CV_32F);
	for(int i = 0; i < m>>1; i++)
	{
		for(int j = 0; j < n>>1; j++)
		{
			depthMap.at<float>(i,j) = Z.at<float>(i,j);
		}
	}
	return depthMap;
}

/**
 * calculate the depth from gradient
 * using Frankot Chellappa Algorithm
 * */
int depthFromGradient( const cv::Mat norm, cv::Mat& depthMap)
{
	// generate p, q matrix
	int matRows = norm.rows, matCols = norm.cols;
	Mat pMat(matRows, matCols, CV_32F), qMat(matRows, matCols, CV_32F);

	for(int h = 0; h < matRows; h++)
	{
		for(int w = 0; w < matCols; w++)
		{
			pMat.at<float>(h, w) = -norm.at<Vec3f>(h, w)[0] / norm.at<Vec3f>(h, w)[2];
			qMat.at<float>(h, w) = -norm.at<Vec3f>(h, w)[1] / norm.at<Vec3f>(h, w)[2];

			if(pMat.at<float>(h, w) != pMat.at<float>(h, w))
				pMat.at<float>(h, w) = 0;

			if(qMat.at<float>(h, w) != qMat.at<float>(h, w))
				qMat.at<float>(h, w) = 0;
		}
	}

	//	depthMap = getGlobalHeights(pMat,qMat);
	depthMap = performFCAlgo(pMat,qMat);

	// recover NAN
	for(int h = 0; h < matRows; h++)
	{
		for(int w = 0; w < matCols; w++)
		{
			if(norm.at<Vec3f>(h,w)[0] != norm.at<Vec3f>(h,w)[0])
			{
				depthMap.at<float>(h, w) = NAN;
			}
		}
	}
	return 1;
}

/**
 * Step 7: dump results.
 * */
int dumpResults(int format,const string filename, const cv::Mat& depthMap)
{
	//TODO: how many formats can we choose ?

	int matRows = depthMap.rows, matCols = depthMap.cols;
	ofstream of(filename.c_str());

	for(int h = 0; h < matRows; h++)
	{
		for(int w = 0; w < matCols; w++)
		{
			if(format == 1)
			{
				// Ascii:
				of << (h + 1) << "," << (w + 1) << "," << depthMap.at<float>(h, w) << "\n";
			} else if(format == 2) {
				of << depthMap.at<float>(h, w)<<" ";
			}
		}

		if(format == 2)
		{
			of  << "\n";
		}
	}
	of.close();
	return 1;
}

int saveCalibrationData(const string filename, const Vec3f* lightsource)
{
	ofstream of(filename.c_str());

	for(int i = 0; i < NUM_IMAGES; i++)
	{
		of << lightsource[i][0] <<" "<<lightsource[i][1] <<" "<<lightsource[i][2] <<" " << "\n";
		cout << lightsource[i][0] <<" "<<lightsource[i][1] <<" "<<lightsource[i][2] <<" " << "\n";
	}
	of.close();
	return 1;
}

int loadCalibrationData(const string filename, Vec3f* lightsource)
{
	ifstream is(filename.c_str());
	//	char buf[100];
	//	while(is.getline(buf, 100)) {
	//		cout <<buf<<endl;
	//	}

	for(int i = 0; i < NUM_IMAGES; i++)
	{
		is >> lightsource[i][0] >> lightsource[i][1]>> lightsource[i][2];
		cout << "Loading Data:" << endl;
		cout << lightsource[i] << endl;
	}
	is.close();
	return 1;
}

int calibrateLightsource(const string filename)
{
	/**
	 * Version 1.0:
	 * I will use all four images to do this calibration.
	 * The circle will be the one that with the smallest fitting error.
	 * */
	float ferrors[NUM_IMAGES], minError = FLT_MAX; // note that the information stored in here is in matrix format
	Vec3f lightsource[NUM_IMAGES], ball;
	Mat filteredimg[NUM_IMAGES];
	int f;
	char buffer[100];
	for( f = 0; f < NUM_IMAGES; f++)
	{
		sprintf(buffer, DATA_FOLDER"Image_0%d - Copy.jpg", (f+1));
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
		filterImage( grayImg, filImag);
		filImag.convertTo(filteredimg[f], CV_32F);
		imwrite( RESULT_FOLDER"blur.jpg", filteredimg[f] );

		// Step 2: locate the ball
		Vec2f center, highlight;
		float radius;

		ferrors[f] = extractHighLight(filImag, radius, highlight, center);

		if(minError > ferrors[f])
		{
			minError = ferrors[f];
			ball[0] = center[0];
			ball[1] = center[1];
			ball[2] = radius;
		}

		cout<< highlight[0] << " " <<highlight[1] <<endl;
		// Step 3: calculate the incident direction
		// TODO: refine the light source properties
		calculateIncidentDir(center, radius, highlight, lightsource[f]);
	}

	for( f = 0; f < NUM_IMAGES; f++)
	{
		// second pass. filter out the wrong cases
		if(abs(minError - ferrors[f]) > CALIBRATE_FIT_ERROR_DIF_THRESHOLD)
		{
			// refine the highlight point
			Vec2f highlight;
			Mat highMask(filteredimg[f].size(), CV_8U, Scalar(0));
			circle( highMask, Point(ball[1], ball[0]), (int)ball[2] + 1, Scalar(255), -1, CV_AA);
			calculateHighligtWithin(filteredimg[f], highMask, highlight);
			calculateIncidentDir(Vec2f(ball[0], ball[1]), ball[2], highlight, lightsource[f]);
		}
	}

	saveCalibrationData(filename, lightsource);

	return 1;
}


int main(int argc, char ** argv)
{
	// first thing first: parse input data
	help(argv[0]);
	int mode = 0;

	Vec3f lightsource[NUM_IMAGES];
	string filename(DATA_FOLDER"lights.txt");

	if(mode == 0)
	{	// load source images and the calibration data
		cout<< "Loading light source data.." <<endl;
		loadCalibrationData(filename, lightsource);
	}else if(mode == 1) {
		calibrateLightsource(filename);
		return 1;
	}
	cout<<"Loading images..."<<endl;
	// open images and blur them
	Mat filteredimg[NUM_IMAGES];
	char buffer[100];
	for( int f = 0; f < NUM_IMAGES; f++)
	{
		sprintf(buffer, DATA_FOLDER"Image_%d.jpg", (f+1));
		// load gray image is enough
		colorImg = imread(buffer);
		if( colorImg.data == NULL) {
			cout<< "Failed to load image." <<endl;
		}
		Mat grayImg, filImag;
		cvtColor(colorImg, grayImg,CV_BGR2GRAY);
		imwrite( RESULT_FOLDER"gray.jpg", grayImg );


		// Step 1: preprocessing
		//filterImage( grayImg, filImag);
		grayImg.convertTo(filteredimg[f], CV_32F);
		imwrite( RESULT_FOLDER"blur.jpg", filteredimg[f] );
	}

	cout<< "Generating shadow masks..." <<endl;
	// Step 4: deal with black pixels and merge matrix.
	cv::Mat shadowMask[NUM_IMAGES];
	processInputImages(shadowMask, filteredimg);

	// Step 5: applying Photometric Stereo
	// TODO: use reflactance map transformation to enhance surface details
	cout<< "Applying photomeric stereo..." <<endl;
	Mat albedo, norm;
	applyPhotometricStereo( shadowMask, filteredimg, lightsource, albedo, norm );

	//	ofstream abfile(DATA_FOLDER"ab.txt");
	//	abfile << albedo<<endl;
	//	abfile.close();

	// Step 6: get depth from norm
	cout<< "Getting depth from gradients..." <<endl;
	Mat depthMap;
	depthFromGradient(norm, depthMap);

	//	ofstream dp(DATA_FOLDER"depth.txt");
	//	dp << depthMap<<endl;
	//	dp.close();

	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Display window", depthMap );
	// Step 7: dump results.
	string output = DATA_FOLDER"output.asc";

	dumpResults(2,output,depthMap);
	cout<< "Done..!" <<endl;
	waitKey();
	return 0;
}
