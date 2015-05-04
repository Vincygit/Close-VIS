/*
 * PhotometricStereoSolver.cpp
 *
 *  Created on: Mar 16, 2015
 *      Author: vincy
 */

#include "PhotometricStereoSolver.h"

using namespace std;
using namespace cv;

namespace ps {

//#define APPLY_FC_ALGORITHM

PhotometricStereoSolver::PhotometricStereoSolver() {
	// TODO Auto-generated constructor stub

}

PhotometricStereoSolver::~PhotometricStereoSolver() {
	// TODO Auto-generated destructor stub
}

/***
 * process the input image.
 * 1. normalize image intensities, get the shadowMask
 * 2. deal with the black pixels.
 * */
int PhotometricStereoSolver::processInputImages(cv::Mat* shadowMask, const cv::Mat* srcImages, float shadowThreshold)
{
	float MaxVal = 255.0f;
	// TODO: apply the percentile method
	//	for(unsigned f = 0; f < numImages; f++)
	//	{
	//		// first pass, find the maximum value
	//		//
	//	}
	float realThresh = MaxVal * shadowThreshold;

	int erosion_type = MORPH_ELLIPSE, erosion_size = 5;

	//	Mat element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ), Point( erosion_size, erosion_size ) );

	// second pass, uniform all pixels
	for(int f = 0; f< numImages; f++)
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
int PhotometricStereoSolver::applyPhotometricStereo(const cv::Mat* shadowMask, const cv::Mat* srcImages, const cv::Vec3f* srcLight, cv::Mat& albedo, cv::Mat& norm)
{
	// first use the shadow mask to filter the srcImage pixels.
	// we only consider pixels that appear more than 3 times >=3
	// the result is merged into a new matrix
	int matRows = srcImages[0].rows, matCols = srcImages[0].cols;

	Mat srcLightMat(numImages, 3, CV_32F); // 3xM
	for(int f = 0; f < numImages; f++)
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
	Mat intensityMat(numImages, totalPoints, CV_32F, Scalar(0));

	for(int f = 0; f < numImages; f++)
	{
		totalMask += shadowMask[f] * (1<<f);
	}

	for(int f = 0; f < numImages; f++)
	{
		for(int h = 0; h < matRows; h++)
		{
			for(int w = 0; w < matCols; w++)
			{
				//				if(totalMask.at<uchar>(h, w) == ((1<<numImages) -1 ))
				//FIXME: why flip this ??
				intensityMat.at<float>(f, h*matCols + w) = srcImages[f].at<float>(h, w) / 255.0;
				//				intensityMat.at<float>(f, h*matCols + w) = srcImages[f].at<float>(h, w) / 255.0;
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
	//	for(int f = 0; f < numImages; f++)
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
	}

	if(norm.data == NULL)
	{
		norm.create(matRows, matCols, CV_32FC3);
	}

	for(int h = 0; h < matRows; h++)
	{
		for(int w = 0; w < matCols; w++)
		{
			int refIdx = h*matCols + w;
			float x = reflectMat.at<float>(0, refIdx);
			float y = reflectMat.at<float>(1, refIdx);
			float z = reflectMat.at<float>(2, refIdx);
			albedo.at<float>(h, w) = sqrt(x*x + y*y + z*z);

			for( int i = 0; i < 3; i++)
			{
				float x = reflectMat.at<float>(i, refIdx);
				norm.at<Vec3f>(h, w)[i] = x / albedo.at<float>(h, w);
			}
		}
	}

	return 1;
}

/**
 * Use this function for the near lighting PS situation.
 * There are mainly two differences:
 * 1. The input light source matrix is different. Now it means the space position of the light source. NOT DIRECTION.
 * 2. The algorithm will run differently. Now for each pixel, the L matrix is different.
 * 3. In this function, we will iteratively solve the function for EACH pixel.
 *
 * see [Uncalibrated Near-Light Photometric Stereo T. Papadhimitri 2014] for more information.
 * */
int PhotometricStereoSolver::applyNearRangePS(cv::Mat& depthMap, const cv::Mat* shadowMask, const cv::Mat* srcImages, const cv::Vec3f* srcLight, cv::Mat& albedo, cv::Mat& norm, int Max_iter)
{
	int matRows = srcImages[0].rows, matCols = srcImages[0].cols;

	// MX3: each row represent a light position.
	Mat srcLightMat(numImages, 3, CV_32F);

	for(int f = 0; f < numImages; f++)
	{
		srcLightMat.at<float>(f,0) = srcLight[f][0];
		srcLightMat.at<float>(f,1) = srcLight[f][1];
		srcLightMat.at<float>(f,2) = srcLight[f][2];
	}

	if(albedo.data == NULL)
	{
		albedo.create(matRows, matCols, CV_32F);
	}

	if(norm.data == NULL)
	{
		norm.create(matRows, matCols, CV_32FC3);
	}

	float iterError = .0f;
	// FIXME FIXME FIXME FIXME
	const float MIN_ERROR = -999999.0f;
	int round_counter = 0;

	// create temporal caching matrix.
	Mat intensityMat(numImages, 1, CV_32F, Scalar(0));
	Mat lightingMat(numImages, 3, CV_32F, Scalar(0));

	// cache the previous calculated depth so as to calculate the error between two rounds.
	Mat preDepth;

	while(iterError > MIN_ERROR && round_counter++ < Max_iter)
	{
		iterError = 0.f;
		// calculate the norm on a pixel basis
		// first, generate the intensity matrix & lighting matrix

		for(int h = 0; h < matRows; h++)
		{
			for(int w = 0; w < matCols; w++)
			{
				// for each pixel
				for(int i = 0; i < numImages; i++)
				{
					Vec3f pxlLighting =  srcLight[i] - Vec3f(h, w, round_counter == 1 ? 1 : depthMap.at<float>(h, w));
					pxlLighting = pxlLighting / (pxlLighting[0]*pxlLighting[0] + pxlLighting[1]*pxlLighting[1] + pxlLighting[2]*pxlLighting[2]);
					// for each dimension
					for(int j = 0; j < 3; j++)
						lightingMat.at<float>(i, j) = pxlLighting[j];

					intensityMat.at<float>(i, 0) = srcImages[i].at<float>(h, w) / 255.0;
				}

				// second, solve the equation for least square    this is a 3x1 matrix
				Mat pxlNorm = lightingMat.inv(DECOMP_SVD) * intensityMat;

				// third, save the parameter back to the norm and albedo matrix
				float mag = .0f;
				for(int c = 0; c < 3; c++) {
					mag += pxlNorm.at<float>(c) *pxlNorm.at<float>(c);
				}

				albedo.at<float>(h, w) = sqrt(mag);
				for(int c = 0; c < 3; c++) {
					norm.at<Vec3f>(h, w)[c] = pxlNorm.at<float>(c) / albedo.at<float>(h, w);
				}

				// calculate the depth error.
				if(round_counter > 1)
					iterError += abs(depthMap.at<float>(h, w) - preDepth.at<float>(h, w));
			}
		}
		iterError /= matRows * matCols;
		cout << "**Iteration: "<<round_counter<<" with mean abs error:" << iterError <<endl;

		if(round_counter > 1)
			depthMap.copyTo(preDepth);

		depthFromGradient( norm, depthMap, 0);
	}

	/*******************************************/
	/** iteration done.!
	/*******************************************/
	cout << "**Iteration DONE! " << endl;
	return 1;
}


void PhotometricStereoSolver::integrate( float *P,  float *Q,  float *Z, int width, int height, float mu)
{
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

//void PhotometricStereoSolver::Vintegrate( Mat &P,  Mat &Q,  Mat&Z, int width, int height, float mu)
//{
//	/* get current i,j position in image */
//	for(int i = 0; i < height; i++)
//	{
//		for(int j = 0; j < width; j++)
//		{
//			if ((i != 0) || (j != 0)) {
//				float v = sin((float)(i*2*M_PI/height));
//				float u = sin((float)(j*2*M_PI/width));
//				float uv = pow(u,2)+pow(v,2);
//				float d = uv + mu*(pow(u,4)+pow(v,4));
//				/* offset = (row * numCols * numChannels) + (col * numChannels) + channel */
//				Z.at<Vec2f>(i, j)[0] = ((u+mu*pow(u,3))*P.at<Vec2f>(i, j)[1]  + (v+mu*pow(v,3))*Q.at<Vec2f>(i, j)[1]) / d;
//				Z.at<Vec2f>(i, j)[1] = (-(u+mu*pow(u,3))*P.at<Vec2f>(i, j)[0] - (v+mu*pow(v,3))*Q.at<Vec2f>(i, j)[0]) / d;
//			}
//		}
//	}
//}

cv::Mat PhotometricStereoSolver::getGlobalHeights(cv::Mat Pgrads, cv::Mat Qgrads)
{

	cv::Mat P(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Q(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));
	cv::Mat Z(Pgrads.rows, Pgrads.cols, CV_32FC2, cv::Scalar::all(0));

	cv::dft(Pgrads, P, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(Qgrads, Q, cv::DFT_COMPLEX_OUTPUT);

	integrate((float *)P.data, (float *)Q.data, (float *)Z.data,Pgrads.cols, Pgrads.rows);
	//	Vintegrate(P, Q, Z,Pgrads.cols, Pgrads.rows);

	/* setting unknown average height to zero */
	Z.at<cv::Vec2f>(0, 0)[0] = 0.0f;
	Z.at<cv::Vec2f>(0, 0)[1] = 0.0f;

	cv::dft(Z, Z, cv::DFT_INVERSE | cv::DFT_SCALE |  cv::DFT_REAL_OUTPUT);

	return Z;
}


/**
 * NOTE: This method is much better!
 * Because we create a periodic matrix rather than the original one.
 * By introducing period in during the DFT, the boudaries have much better performance.
 * **/
cv::Mat PhotometricStereoSolver::performFCAlgo(const cv::Mat&Pmat, const cv::Mat&Qmat, float lambda)
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
				// TODO: Note this change!!  we this is right!
				float u = vMat.at<float>(i,j)/m;
				float v = uMat.at<float>(i,j)/n;

#ifdef APPLY_FC_ALGORITHM
				double d = 2*M_PI*( pow(u, 2 ) + pow(v, 2 ) );
				Z.at<Vec2f>(i, j)[0] = (u*P.at<Vec2f>(i,j)[1] + v*Q.at<Vec2f>(i,j)[1]) / d;
				Z.at<Vec2f>(i, j)[1] = -(u*P.at<Vec2f>(i,j)[0] + v*Q.at<Vec2f>(i,j)[0]) / d;
#else
				double d = 2*M_PI*( pow( u, 2 ) + pow( v, 2 ) + lambda * (pow( u, 4 ) + pow( v, 4 )) );
				Z.at<Vec2f>(i, j)[0] = ((u + lambda * pow(u,3))*P.at<Vec2f>(i,j)[1] + (v + lambda * pow(v,3))*Q.at<Vec2f>(i,j)[1]) / d;
				Z.at<Vec2f>(i, j)[1] = -((u + lambda * pow(u,3))*P.at<Vec2f>(i,j)[0] + (v + lambda * pow(v,3))*Q.at<Vec2f>(i,j)[0]) / d;
				//				float u = sin((float)(i*2*M_PI/m));
				//				float v = sin((float)(j*2*M_PI/n));
				//				float mu = 0.5;
				//				float uv = pow(u,2)+pow(v,2);
				//				float d = uv + mu*(pow(u,4)+pow(v,4));
				//				/* offset = (row * numCols * numChannels) + (col * numChannels) + channel */
				//				Z.at<Vec2f>(i, j)[0] = ((u+mu*pow(u,3))*P.at<Vec2f>(i, j)[1]  + (v+mu*pow(v,3))*Q.at<Vec2f>(i, j)[1]) / d;
				//				Z.at<Vec2f>(i, j)[1] = (-(u+mu*pow(u,3))*P.at<Vec2f>(i, j)[0] - (v+mu*pow(v,3))*Q.at<Vec2f>(i, j)[0]) / d;
#endif
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
int PhotometricStereoSolver::depthFromGradient( const cv::Mat norm, cv::Mat& depthMap, int method)
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
	//TODO:: BINBIN
	//	method=1;
	switch(method) {
	case 1:
		// use new solving matrix by vincy
		depthMap = getGlobalHeights(pMat,qMat);
		break;

	default:
		// use Ying Xiong's method from Matlab, this method may introduce curvature.
		depthMap = performFCAlgo(pMat,qMat);
		break;
	}

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
int PhotometricStereoSolver::dumpResults(int format,const string filename, const cv::Mat& depthMap)
{
	//TODO: how many formats can we choose ?

	int matRows = depthMap.rows, matCols = depthMap.cols;
	ofstream of(filename.c_str());

	// re scale the depth map
	Mat scaledMap(depthMap.size(), CV_32F);
	double min = FLT_MAX, max=-FLT_MAX;
	for(int h = 0; h < depthMap.rows; h++)
	{
		for(int w = 0; w < depthMap.cols; w++)
		{
			if(min > depthMap.at<float>(h,w))
			{
				min = depthMap.at<float>(h,w);
			}

			if(max < depthMap.at<float>(h,w))
			{
				max = depthMap.at<float>(h,w);
			}
		}
	}
	scaledMap = depthMap - min;
	//	scaledMap.convertTo(scaledMap, CV_32F, 1.0 / (max - min));

	for(int h = 0; h < matRows; h++)
	{
		for(int w = 0; w < matCols; w++)
		{
			if(format == 1)
			{
				// obj format
				of << "v "<<(h + 1) << " " << (w + 1) << " " << scaledMap.at<float>(h, w) << "\n";
			} else if(format == 2) {
				of << depthMap.at<float>(h, w)<<" ";
			}
		}

		if(format == 2)
		{
			of  << "\n";
		}
	}


	if(format == 1)
	{
		for(int h = 0; h < matRows; h++)
		{
			for(int w = 0; w < matCols; w++)
			{
				int curIdx = h * matCols + w + 1;
				if(w > 0 && h < matRows - 1)
				{
					of << "f "<< curIdx << " " << (curIdx + matCols - 1)<< " " << (curIdx + matCols)  << "\n";
				}

				if(w < matCols - 1 && h < matRows - 1)
				{
					of << "f "<< curIdx << " " << (curIdx + matCols) << " " << (curIdx + 1) << "\n";
				}
			}

		}
	}
	of.close();
	return 1;
}

} /* namespace std */
