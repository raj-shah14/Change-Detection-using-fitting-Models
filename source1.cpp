#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <stdlib.h>
#include <stdio.h>
#include<opencv2/opencv.hpp>
#include<numeric>
#include<iostream>
#include<math.h>

using namespace cv;
using namespace std;


cv::RotatedRect getErrorEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat);

Mat img1, img2,src1,src2,src3,src4,src5,src6, plot;
double mean,thresholdval,cthresholdval,radius;
int i, j, k, n, choseImgSet, selectType, distType,gaus;
double array1[1000000] = {};
double array2[1000000] = {};
vector<Point2f>intensity;
vector<Point2f>location;
Vec4f lines;

int main(int argc, char** argv) {
/*	if (argc != 7)
	{
		cout << "Error" << endl;
		return -1;
	}
	Mat src1 = imread(argv[1], 0);
	Mat src2 = imread(argv[2], 0);
	Mat src3 = imread(argv[3], 0);
	Mat src4 = imread(argv[4], 0);
	Mat src5 = imread(argv[5], 0);
	Mat src6 = imread(argv[6], 0);
*/
	Mat src1 = imread("1-in000001.jpg", 0);
	Mat src2 = imread("1-in001148.jpg", 0);
	Mat src3 = imread("2-in000001.jpg", 0);
	Mat src4 = imread("2-in001800.jpg", 0);
	Mat src5 = imread("3-in000001.jpg", 0);
	Mat src6 = imread("3-in000500.jpg", 0);
	
	cout << "Enter 1, 2 or 3 for image set to be selected" << endl;
	cin >> choseImgSet;
	cout << "Enter type of line fit: 1--> LeastSquare , 2--> Ransac" << endl;
	cin >> selectType;
	if (choseImgSet == 1) {
		src1.copyTo(img1);
		src2.copyTo(img2);
		if (selectType == 1) {
			distType = 2;
			thresholdval = 25;
			cthresholdval = 100;
		}
		if (selectType == 2) {
			distType = 6;
			thresholdval = 25;
			cthresholdval = 100;
		}
	}
	if (choseImgSet == 2) {
		src3.copyTo(img1);
		src4.copyTo(img2);
		if (selectType == 1) {
			distType = 2;
			thresholdval = 20;
			cthresholdval = 100;
		}
		if (selectType == 2) {
			distType = 6;
			thresholdval = 27;
			cthresholdval = 100;
		}
	}
	if (choseImgSet == 3) {
		src5.copyTo(img1);
		src6.copyTo(img2);
		if (selectType == 1) {
			distType = 2;
			thresholdval = 60;
			cthresholdval = 100;
		}
		if (selectType == 2) {
			distType = 6;
			thresholdval = 60;
			cthresholdval = 100;
		}
	}
	
	cout << "Do you want Gaussian as well? 1--> Yes , 2 --> No" << endl;
	cin >> gaus;

	imshow("Image-1", img1);
	moveWindow("Image-1", 0, 20);
	imshow("Image-2", img2);
	moveWindow("Image-2", 350, 20);

	//CREATING THE PLOT
	Mat plot(256, 256, CV_8UC3, Scalar(0, 0, 0));
	for (i = 0; i < img1.rows; i++) {
		for (j = 0; j < img1.cols; j++) {
			array1[(i*img1.cols) + j] = (int)img1.at<uchar>(i, j);
			array2[(i*img2.cols) + j] = (int)img2.at<uchar>(i, j);
			
			plot.at<Vec3b>(array1[(i*img1.cols) + j], array2[(i*img2.cols) + j])[0] = 255;
			intensity.push_back(Point(img1.at<uchar>(i, j), img2.at<uchar>(i, j)));
			location.push_back(Point(i, j));
		}
	}

	imshow("Plot", plot);
	moveWindow("Plot", 600, 20);
		

	Mat change = plot.clone();
	Mat ellithresh = plot.clone();
	Mat mean = plot.clone();
	Mat circlethresh = plot.clone();
	Mat plotellipse = plot.clone();

	
	//FITTING THE LINE TO THE POINTS
	fitLine(intensity, lines, distType, 0, 0.01, 0.01);

	int lefty = (-lines[2] * lines[1] / lines[0]) + lines[3];
	int righty = ((plot.cols - lines[2])*lines[1] / lines[0]) + lines[3];


	//PLOTTING THE LINE 
	line(plot, Point(plot.cols - 1, righty), Point(0, lefty), Scalar(0, 0, 255), 2);
	imshow("Fitline", plot);
	moveWindow("Fitline", 900, 20);



	CvPoint* point = new CvPoint();

	//COMPUTING MEAN AND SETTING THRESHOLDING DISTANCE
	double slope = (double)(lefty - righty) / 255;  //SLOPE OF THE LINE
	float angleline = cvFastArctan(-(float)(lefty - righty), 255);
	cout << "Angle=" << angleline << ",slope=" << slope << endl;
	double meanx = 0;
	double meany = 0;
	for (i = 0; i < img1.rows; i++) {
		for (j = 0; j < img1.cols; j++) {
			point->x = (int)img1.at<uchar>(i, j);
			point->y = (int)img2.at<uchar>(i, j);
			meanx += (point[0].x);
			meany += (point[0].y);
			double distance = abs(((righty - lefty)*point[0].y) - (plot.cols*point[0].x) + (plot.cols * lefty)) / sqrt((righty - lefty)*(righty - lefty) + (plot.cols * plot.cols));

			//SETTING THRESHOLDING DISTANCE											//FOR CV_DIST_WELSCH IMG_SET1 thresh (distance)<27    IMG-SET2 thresh (distance)<27   IMG-SET3 thresh (distance)<60
			if (distance < thresholdval) {											//FOR CV_DIST_L2 IMG-SET1 thresh DISTANCE<20    IMG-SET2 thresh DISTANCE<20   IMG-SET3 thresh DISTANCE<60
				array1[(i*img1.cols) + j] = (int)img1.at<uchar>(i, j);
				array2[(i*img2.cols) + j] = (int)img2.at<uchar>(i, j);
				change.at<Vec3b>(array1[(i*img1.cols) + j], array2[(i*img2.cols) + j])[0] = 0;
			}
		}
	}


	//PLOTTING THE THRESHOLD
	imshow("Threshold", change);
	moveWindow("Threshold", 1200, 20);
	cout << "Processing Please wait...." << endl;
	
	//PLOTTING THE VALUE OF REMAINING PIXELS
	Mat detect = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
	for (i = 0; i < change.rows; i++) {
		for (j = 0; j < change.cols; j++) {
			if ((int)change.at<Vec3b>(i, j)[0] == 255) {
				for (k = 0; k < img1.rows; k++) {
					for (n = 0; n < img1.cols; n++) {
						if (array1[(k*img1.cols) + n] == i && array2[(k*img2.cols) + n] == j) {
							detect.at<uchar>(k, n) = i;
							detect.at<uchar>(k, n) = j;

						}
					}
				}
				//cout << i << "," << j<< endl;
			}
		}
	}

	cout << "Done" << endl;
	imshow("ChangeDetectionFitline", detect);
	moveWindow("ChangeDetectionFitline", 0, 400);
	
	//ERODING THE IMAGE
	Mat ero;
	erode(detect, ero, Mat(), Point(-1, -1), 1);
	imshow("FitLine Erode", ero);
	moveWindow("FitLine Erode", 350, 400);
	
	if (gaus == 1) {
		//CALCULATING MEAN FOR GAUSSIAN
		CvPoint* result = new CvPoint();
		result->x = meanx / (img1.rows*img1.cols);
		result->y = meany / (img2.rows*img2.cols);


		//CALCULATING THE COVARIANCE
		double covarxy = 0, covarx = 0, covary = 0;
		for (i = 0; i < img1.rows; i++) {
			for (j = 0; j < img1.cols; j++) {
				covarx += (((int)img1.at<uchar>(i, j) - result[0].x)*((int)img1.at<uchar>(i, j) - result[0].x));
				covarxy += (((int)img1.at<uchar>(i, j) - result[0].x)*((int)img2.at<uchar>(i, j) - result[0].y));
				covary += (((int)img2.at<uchar>(i, j) - result[0].y)*((int)img2.at<uchar>(i, j) - result[0].y));
			}
		}

		//CALCULATING GAUSSIAN PARAMETERS
		covarx = covarx / (img1.rows*img1.cols);
		covarxy = covarxy / (img1.rows*img1.cols);
		covary = covary / (img2.rows*img2.cols);

		//CIRCLE THRESHOLDING
		circle(circlethresh, Point(result[0].x, result[0].y), cthresholdval, Scalar(0,255, 255), 1, 8, 0);
		circle(circlethresh, Point(result[0].x, result[0].y), 2, Scalar(0, 255, 255), 1, 8, 0);
		imshow("Distance-Plot",circlethresh);
		circle(circlethresh, Point(result[0].x, result[0].y), cthresholdval, Scalar(0, 0, 0), 1, 8, 0);
		circle(circlethresh, Point(result[0].x, result[0].y), 2, Scalar(0, 0, 0), 1, 8, 0);

		for (i = 0; i < img1.rows; i++) {
			for (j = 0; j < img1.cols; j++) {
				radius = (((int)img1.at<uchar>(i, j) - result[0].x)*((int)img1.at<uchar>(i, j) - result[0].x)) + (((int)img2.at<uchar>(i, j) - result[0].y)*((int)img2.at<uchar>(i, j) - result[0].y));
				if (radius <= (cthresholdval*cthresholdval)) {
					circlethresh.at<Vec3b>((int)img1.at<uchar>(i, j), (int)img2.at<uchar>(i, j))[0] = 0;
				}
			}
		}

		imshow("Distance Threshold", circlethresh);

		//RETRIEVING THE IMAGE FROM CIRCLE

		Mat distdetect = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
		for (i = 0; i < circlethresh.rows; i++) {
			for (j = 0; j < circlethresh.cols; j++) {
				if ((int)circlethresh.at<Vec3b>(i, j)[0] == 255) {
					for (k = 0; k < img1.rows; k++) {
						for (n = 0; n < img1.cols; n++) {
							if (array1[(k*img1.cols) + n] == i && array2[(k*img2.cols) + n] == j) {
								distdetect.at<uchar>(k, n) = i;
								distdetect.at<uchar>(k, n) = j;

							}
						}
					}
					//cout << i << "," << j<< endl;
				}
			}
		}
		
		imshow("DistanceChangeDetection", distdetect);
		Mat cero;
		erode(detect, cero, Mat(), Point(-1, -1), 1);
		imshow("Distance Erode", cero);


		//PLOTTING THE MEAN POINT
		circle(plotellipse, Point(result[0].x, result[0].y), 2, Scalar(255, 255, 255), 1, 8, 0);
		cout << "Mean x=" << result[0].x << ",mean y=" << result[0].y << endl;
		cout << "Covarx=" << (covarx) << ",Covary=" << (covary) << ",covarxy=" << (covarxy) << endl;

		//DRAWING THE ELLIPSE
		Mat covmat = (Mat_<double>(2, 2) << (covarx), (covarxy), (covarxy), (covary));
		Point2f centrepoint(result[0].x, result[0].y);
		RotatedRect ellipse = getErrorEllipse(6, centrepoint, covmat);
		cv::ellipse(plotellipse, ellipse, Scalar(0, 255, 0), 2, 8);


		//DISPLAYING THE FINAL IMAGE
		imshow("Ellipse", plotellipse);
		moveWindow("Ellipse", 700, 400);
		float ellipseangle = (ellipse.angle*3.14159265359) / 180;

		//ORIENTING AND THRESHOLDING THE ELLIPSE
		for (i = 0; i < img1.rows; i++) {
			for (j = 0; j < img1.cols; j++) {
				double first = ((((int)img1.at<uchar>(i, j) - result[0].x)*cos(ellipseangle) + ((int)img2.at<uchar>(i, j) - result[0].y)*sin(ellipseangle))*(((int)img1.at<uchar>(i, j) - result[0].x)*cos(ellipseangle) + ((int)img2.at<uchar>(i, j) - result[0].y)*sin(ellipseangle))) / ((ellipse.size.width / 2)*(ellipse.size.width / 2));
				double second = ((((int)img1.at<uchar>(i, j) - result[0].x)*sin(ellipseangle) - ((int)img2.at<uchar>(i, j) - result[0].y)*cos(ellipseangle))*(((int)img1.at<uchar>(i, j) - result[0].x)*sin(ellipseangle) - ((int)img2.at<uchar>(i, j) - result[0].y)*cos(ellipseangle))) / ((ellipse.size.height / 2)*(ellipse.size.height / 2));
				double ellidist = first + second;
				if (ellidist <= 1) {
					ellithresh.at<Vec3b>((int)img1.at<uchar>(i, j), (int)img2.at<uchar>(i, j))[0] = 0;
				}
			}
		}


		imshow("Thresholdellipse", ellithresh);
		moveWindow("Thresholdellipse", 1000, 400);

		cout << "Processing Please wait...." << endl;
		Mat gaussdetect = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
		for (i = 0; i < ellithresh.rows; i++) {
			for (j = 0; j < ellithresh.cols; j++) {
				if ((int)ellithresh.at<Vec3b>(i, j)[0] == 255) {
					for (k = 0; k < img1.rows; k++) {
						for (n = 0; n < img1.cols; n++) {
							if (array1[(k*img1.cols) + n] == i && array2[(k*img2.cols) + n] == j) {
								gaussdetect.at<uchar>(k, n) = i;
								gaussdetect.at<uchar>(k, n) = j;

							}
						}
					}
				}
			}
		}
		
		//system("CLS");
		imshow("GaussianChangeDetection", gaussdetect);
		moveWindow("GaussianChangeDetection", 1150, 400);
		Mat gero;
		erode(gaussdetect, gero, Mat(), Point(-1, -1), 1);
		imshow("Gaussian Eroded", gero);
		cout << "Done" << endl;
	}
	waitKey(0);
	return 0;
}


RotatedRect getErrorEllipse(double chisquare_val, Point2f mean, Mat covmat) {

	//Get the eigenvalues and eigenvectors
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(covmat, eigenvalues, eigenvectors);

	//Calculate the angle between the largest eigenvector and the x-axis
	double angle = atan2(eigenvectors.at<double>(0, 1), eigenvectors.at<double>(0, 0));

	//Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
	if (angle < 0)
		angle += 6.28318530718;

	//Conver to degrees instead of radians
	angle = 180 * angle / 3.14159265359;

	//Calculate the size of the minor and major axes
	double majoraxissize = chisquare_val*sqrt(eigenvalues.at<double>(0));
	double minoraxissize = chisquare_val*sqrt(eigenvalues.at<double>(1));

	cout << "majaxis=" << majoraxissize << ",minaxis=" << minoraxissize << ",angle=" << angle << endl;

	//Return the oriented ellipse
	//The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
	return RotatedRect(mean, Size2f(majoraxissize, minoraxissize), angle);

}
