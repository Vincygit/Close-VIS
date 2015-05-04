
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include "BestFit.h"

#include "Utils.h"
#include "Calibrator.h"
#include "ConfReader.h"
#include "PhotometricStereoSolver.h"

using namespace std;
using namespace cv;
using namespace ps;

////////////////////////////////
// parameter keys///////////////
////////////////////////////////
#define DataFolder 		"DataFolder"
#define ResultFolder	"ResultFolder"
#define CalibPrefix		"CalibrationPreFix"
#define	DataImgPrefix	"DataImagePrefix"
#define ImageSuffix		"ImageSuffix"

#define NumLights		"NumLights"
#define Mode			"Mode"
#define PSMode			"PSmode"
#define RescaleImg		"ResaleImg"
#define	RescaleWidth	"RescaleWidth"
#define RescaleHeight	"RescaleHeight"
#define OutputFormat	"OutputFormat"

#define FitErrorThresh	"FitErrorThresh"
#define HighlightRadius	"HighlightRadius"
#define	HighLightThresh	"HighLightThresh"
#define DarkLightThresh	"DarkLightThresh"
#define	FitCandidateNum	"FitCandidateNum"
#define	CenterParam		"CenterParam"

///////////////////////////////
/// output file names
///////////////////////////////
#define OutputFile 		"output.obj"
#define	Lights			"lights.txt"
#define Gray			"gray.jpg"


///////////////////////////////
/// Working Mode
///////////////////////////////
#define DoCalibration 1
#define DoPS 2
#define DoInDistantModel 1
#define DoInNearRangeModel 2

/////////////////////////////////////

typedef struct psparam {
	const char* dataFolder;
	const char* resultFolder;
	const char* calibPrefix;
	const char* dataImgPrefix;
	const char* imgSuffix;

	int numLights;
	int mode;
	int PSmode;
	string rescaleImg;
	int rescaleWidth;
	int rescaleHeight;
	string outputformat;

	psparam() {
		dataFolder = NULL;
		resultFolder = NULL;
		PSmode = DoInDistantModel;
	}
} PSParam;

static inline void PrntParamError(string parm) {
	cout<<"Failed to read parameter: ["<<parm<<"]"<<endl;
	exit(-1);
}

// load and package PS parameters
PSParam pkgPSParam(ConfReader &cr) {
	PSParam psparam;
	string buf;

	if(cr.GetParamValue(DataFolder, buf) == -1)
		PrntParamError(DataFolder);
	psparam.dataFolder = buf.c_str();

	if(cr.GetParamValue(ResultFolder, buf) == -1)
		PrntParamError(ResultFolder);
	psparam.resultFolder = buf.c_str();

	if(cr.GetParamValue(CalibPrefix, buf) == -1)
		PrntParamError(CalibPrefix);
	psparam.calibPrefix = buf.c_str();

	if(cr.GetParamValue(DataImgPrefix, buf) == -1)
		PrntParamError(DataImgPrefix);
	psparam.dataImgPrefix = buf.c_str();

	if(cr.GetParamValue(ImageSuffix, buf) == -1)
		PrntParamError(ImageSuffix);
	psparam.imgSuffix = buf.c_str();

	////////////////////////////////////
	if(cr.GetParamValue(NumLights, buf) == -1)
		PrntParamError(NumLights);
	psparam.numLights = atoi(buf.c_str());

	if(cr.GetParamValue(Mode, buf) == -1)
		PrntParamError(Mode);
	psparam.mode = atoi(buf.c_str());

	if(cr.GetParamValue(PSMode, buf) == -1)
		PrntParamError(PSMode);
	psparam.PSmode = atoi(buf.c_str());

	if(cr.GetParamValue(RescaleImg, psparam.rescaleImg) == -1)
		PrntParamError(RescaleImg);

	if(cr.GetParamValue(RescaleWidth, buf) == -1)
		PrntParamError(RescaleWidth);
	psparam.rescaleWidth = atoi(buf.c_str());

	if(cr.GetParamValue(RescaleHeight, buf) == -1)
		PrntParamError(RescaleHeight);
	psparam.rescaleHeight = atoi(buf.c_str());

	if(cr.GetParamValue(OutputFormat, psparam.outputformat) == -1)
		PrntParamError(OutputFormat);

	return psparam;
}
// Note that if you don't use & then the cr here is empty.. Why?
CalibParam pkgCalibratorParam(ConfReader &cr) {
	CalibParam cparam;

	string buf;

	if(cr.GetParamValue(DataFolder, buf) == -1)
		PrntParamError(DataFolder);
	cparam.dataFolder = buf.c_str();

	if(cr.GetParamValue(ResultFolder, buf) == -1)
		PrntParamError(ResultFolder);
	cparam.resultFolder = buf.c_str();

	if(cr.GetParamValue(CalibPrefix, buf) == -1)
		PrntParamError(CalibPrefix);
	cparam.calibPrefix = buf.c_str();

	if(cr.GetParamValue(DataImgPrefix, buf) == -1)
		PrntParamError(DataImgPrefix);
	cparam.dataImgPrefix = buf.c_str();

	if(cr.GetParamValue(ImageSuffix, buf) == -1)
		PrntParamError(ImageSuffix);
	cparam.imgSuffix = buf.c_str();
	/////////////////////////////////////////////
	if(cr.GetParamValue(NumLights, buf) == -1)
		PrntParamError(NumLights);
	cparam.numLights = atoi(buf.c_str());

	if(cr.GetParamValue(HighlightRadius, buf) == -1)
		PrntParamError(HighlightRadius);
	cparam.highLightRaduis = atoi(buf.c_str());

	if(cr.GetParamValue(HighLightThresh, buf) == -1)
		PrntParamError(HighLightThresh);
	cparam.highLightsThreshold = atoi(buf.c_str());

	if(cr.GetParamValue(DarkLightThresh, buf) == -1)
		PrntParamError(DarkLightThresh);
	cparam.darkLightsThreshold = atoi(buf.c_str());

	if(cr.GetParamValue(FitCandidateNum, buf) == -1)
		PrntParamError(FitCandidateNum);
	cparam.fitCandidateNum = atoi(buf.c_str());

	if(cr.GetParamValue(FitErrorThresh, buf) == -1)
		PrntParamError(FitErrorThresh);
	cparam.fitErrorThreshold = atoi(buf.c_str());

	if(cr.GetParamValue(CenterParam, buf) == -1)
		PrntParamError(CenterParam);
	cparam.centerParam = atoi(buf.c_str());

	return cparam;
}

int main(int argc, char ** argv)
{
	// read configure file:
	//	ConfReader cr(argv[1]);
	ConfReader cr("/home/vincy/ps.conf");
	PSParam param = pkgPSParam(cr);
	CalibParam cparam = pkgCalibratorParam(cr);

	// first thing first: parse input data
	help(argv[0]);

	Vec3f lightsource[param.numLights];
	string filename = param.dataFolder + string(Lights);

	PhotometricStereoSolver pssolver(param.numLights);
	Calibrator caliber(cparam);

	if(param.rescaleImg == "yes")
	{
		cout << "rescaling images.."<< endl;
		char buffer[100];
		Mat colorImg;// cropped;
		for(int i = 0; i< param.numLights; i++)
		{
			sprintf(buffer, "%sImage_%d.JPG", param.dataFolder, (i+1));
			cout<<buffer<<endl;
			colorImg = imread(buffer);
			namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
				//	imwrite( string(param.dataFolder + "albedo.jpg"), albedo );

//			imshow("Display window", colorImg);
//			waitKey();
//			rescaleImage(colorImg, param.rescaleHeight, param.rescaleWidth);
//			imwrite(buffer, colorImg);

			//			cropImage(colorImg, cropped, 4);
			//			imwrite(buffer, cropped);

			sprintf(buffer, "%s%s1_%d%s", param.dataFolder, param.calibPrefix, (i+1), param.imgSuffix);
			cout<<buffer<<endl;
			colorImg = imread(buffer);
			rescaleImage(colorImg, param.rescaleHeight, param.rescaleWidth);
			imwrite(buffer, colorImg);

			if(param.PSmode == DoInNearRangeModel) {
				// if in near range model, then we have an extra set of data.
				sprintf(buffer, "%s%s2_%d%s", param.dataFolder, param.calibPrefix, (i+1), param.imgSuffix);
				colorImg = imread(buffer);
				rescaleImage(colorImg, param.rescaleHeight, param.rescaleWidth);
				imwrite(buffer, colorImg);
			}
		}
		cout << "rescaling images done.."<< endl;
	}

	switch(param.mode) {
	case DoCalibration:
		// this is the calibration mode.
		if(param.PSmode == DoInDistantModel) {
			// distant lighting model.
			caliber.calibrateLightsource(filename, param.numLights);
		} else if(param.PSmode == DoInNearRangeModel) {
			caliber.estimateLightPositions(filename, param.numLights);
		}
		return 1;

	case DoPS:
		// this is the PS working mode. We need to load the data and go.
		caliber.loadCalibrationData(filename, lightsource);
		break;
	}

	cout<<"Loading images..."<<endl;
	// open images and blur them
	Mat filteredimg[param.numLights], colorImg;
	char buffer[100];
	for( int f = 0; f < param.numLights; f++)
	{
		sprintf(buffer, "%s%s%d%s", param.dataFolder, param.dataImgPrefix,(f+1), param.imgSuffix);
		// load gray image is enough
		colorImg = imread(buffer);
		if( colorImg.data == NULL) {
			cout<< "Failed to load image." <<endl;
			//TODO exception handler
		}

		Mat grayImg, filImag;
		cvtColor(colorImg, grayImg,CV_BGR2GRAY);
		string grayimgfile = param.resultFolder + string(Gray);
		imwrite( grayimgfile, grayImg );

		// Step 1: preprocessing
		//filterImage( grayImg, filImag);
		grayImg.convertTo(filteredimg[f], CV_32F);
		//		imwrite( RESULT_FOLDER"blur.jpg", filteredimg[f] );
	}

	cout<< "Generating shadow masks..." <<endl;
	// Step 4: deal with black pixels and merge matrix.
	cv::Mat shadowMask[param.numLights];
	pssolver.processInputImages(shadowMask, filteredimg);

	// Step 5: applying Photometric Stereo
	// TODO: use reflactance map transformation to enhance surface details
	cout<< "Applying photometric stereo..." <<endl;
	Mat albedo, norm, depthMap;
	switch(param.PSmode) {
	case DoInDistantModel:	/// use distant PS model.
		pssolver.applyPhotometricStereo( shadowMask, filteredimg, lightsource, albedo, norm );

		cout<< "Getting depth from gradients..." <<endl;
		pssolver.depthFromGradient(norm, depthMap, 0);
		break;
	case DoInNearRangeModel:	/// use near range PS model.
		pssolver.applyNearRangePS(depthMap,shadowMask, filteredimg, lightsource, albedo, norm, 1 );
		break;
	}

	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	//	imwrite( string(param.dataFolder + "albedo.jpg"), albedo );

	imshow("Display window", norm);

	//	imshow( "Display window", depthMap );
	// Step 7: dump results.
	string output = param.dataFolder + string("output.obj");

	pssolver.dumpResults(param.outputformat, output, depthMap);
	cout<< "Done..!" <<endl;

	waitKey();
	return 0;
}
