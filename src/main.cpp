
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include "BestFit.h"

#include "Utils.h"
#include "Calibrator.h"
#include "PhotometricStereoSolver.h"

using namespace std;
using namespace cv;
using namespace ps;

#define NUM_IMAGES 8
#define SCALED_HEIGHT 1000

int amain(int argc, char ** argv)
{
	// 0: photometric stereo
	// 1: calibration
	int mode = 0;
	int rescale = 1;
	int outputformat = 1; // 1: depth matrix, 2: OBJ
	int depthCalcMode = 1 ; // 1 for vincy

	// first thing first: parse input data
	help(argv[0]);

	Vec3f lightsource[NUM_IMAGES];
	string filename(DATA_FOLDER"lights.txt");

	PhotometricStereoSolver pssolver(NUM_IMAGES);
	Calibrator caliber(NUM_IMAGES);

	if(rescale)
	{
		cout << "rescaling images.."<< endl;
		char buffer[100];
		Mat colorImg, cropped;
		for(int i = 0; i< NUM_IMAGES; i++)
		{
			sprintf(buffer, DATA_FOLDER"Image_%d.JPG", (i+1));
			colorImg = imread(buffer);
			rescaleImage(colorImg, SCALED_HEIGHT, SCALED_HEIGHT);
			imwrite(buffer, colorImg);
//			cropImage(colorImg, cropped, 4);
//			imwrite(buffer, cropped);

			sprintf(buffer, DATA_FOLDER"IMG_%d.JPG", (i+1));
			colorImg = imread(buffer);
			rescaleImage(colorImg, SCALED_HEIGHT, SCALED_HEIGHT);
			imwrite(buffer, colorImg);
//			cropImage(colorImg, cropped, 4);
//			imwrite(buffer, cropped);
		}
		cout << "rescaling images done.."<< endl;
	}

	if(mode == 0)
	{	// load source images and the calibration data
		cout<< "Loading light source data.." <<endl;
		caliber.loadCalibrationData(filename, lightsource);
	} else if(mode == 1) {
		caliber.calibrateLightsource(filename, NUM_IMAGES);
//		caliber.estimateLightPositions(filename, NUM_IMAGES);
		return 1;
	}

	cout<<"Loading images..."<<endl;
	// open images and blur them
	Mat filteredimg[NUM_IMAGES], colorImg;
	char buffer[100];
	for( int f = 0; f < NUM_IMAGES; f++)
	{
		sprintf(buffer, DATA_FOLDER"Image_%d.JPG", (f+1));
		// load gray image is enough
		colorImg = imread(buffer);
		if( colorImg.data == NULL) {
			cout<< "Failed to load image." <<endl;
			//TODO exception handler
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
	pssolver.processInputImages(shadowMask, filteredimg);

	// Step 5: applying Photometric Stereo
	// TODO: use reflactance map transformation to enhance surface details
	cout<< "Applying photometric stereo..." <<endl;
	Mat albedo, norm;
//	pssolver.applyPhotometricStereo( shadowMask, filteredimg, lightsource, albedo, norm );
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	imwrite( DATA_FOLDER"albedo.jpg", albedo );
	//	ofstream abfile(DATA_FOLDER"ab.txt");
	//	abfile << albedo<<endl;
	//	abfile.close();

	// Step 6: get depth from norm
	cout<< "Getting depth from gradients..." <<endl;
	Mat depthMap;
//	pssolver.depthFromGradient(norm, depthMap,depthCalcMode);

	pssolver.applyNearRangePS(depthMap,shadowMask, filteredimg, lightsource, albedo, norm, 1 );
	imshow("Display window", norm);
	//	ofstream dp(DATA_FOLDER"depth.txt");
	//	dp << depthMap<<endl;
	//	dp.close();

	//	imshow( "Display window", depthMap );
	// Step 7: dump results.
	string output = DATA_FOLDER"output.obj";

	pssolver.dumpResults(outputformat,output,depthMap);
	cout<< "Done..!" <<endl;
	waitKey();
	return 0;
}
