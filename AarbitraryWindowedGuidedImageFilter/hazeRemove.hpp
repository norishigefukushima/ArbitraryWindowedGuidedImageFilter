#pragma once
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "arbitraryWindowedGuidedImageFilter.hpp"
using namespace cv;
using namespace std;

void minFilter(InputArray src, OutputArray dest, Size kernelSize, int shape= cv::MORPH_RECT)
{
	Mat element = getStructuringElement(shape, kernelSize);
	erode(src, dest, element);
}

void minFilter(InputArray src, OutputArray dest, int radius)
{
	minFilter(src, dest, Size(2 * radius + 1, 2 * radius + 1));
}

class HazeRemove
{
	int filter_switch;
	float sigma_range;
public:
	cv::Size size;
	cv::Mat dark;
	std::vector<cv::Mat> minvalue;
	cv::Mat tmap;
	cv::Scalar A;

	void darkChannel(cv::Mat& src, int r);
	void getAtmosphericLight(cv::Mat& srcImage, double topPercent = 0.1);
	void getTransmissionMap(float omega = 0.95f);
	void removeHaze(cv::Mat& src, cv::Mat& trans, cv::Scalar v, cv::Mat& dest, float clip = 0.3f);
	HazeRemove();
	~HazeRemove();

	void getAtmosphericLightImage(cv::Mat& dest);
	void showTransmissionMap(cv::Mat& dest, bool isPseudoColor = false);
	void showDarkChannel(cv::Mat& dest, bool isPseudoColor = false);
	void operator() (cv::Mat& src, cv::Mat& dest, int r_dark, double toprate, int r_joint, double e_joint);
	void run(cv::Mat& src, std::string wname = "hazeRemove");
};

void HazeRemove::darkChannel(Mat& src, int r)
{
	if (dark.empty())dark.create(size, CV_8U);

	vector<Mat> srcv;
	split(src, srcv);

	for (int i = 0; i < src.channels(); i++)
	{
		minFilter(srcv[i], minvalue[i], r);
	}

	if (src.channels() == 3)
	{
		//#pragma omp parallel for
		for (int i = 0; i < size.height; i++)
		{
			const uchar* sr = minvalue[0].ptr(i);
			const uchar* sg = minvalue[0].ptr(i);
			const uchar* sb = minvalue[0].ptr(i);
			uchar* dptr = dark.ptr(i);

			for (int j = 0; j < size.width; j++)
			{
				uchar minv = min(sr[j], sg[j]);
				dptr[j] = min(minv, sb[j]);
			}
		}
	}
}

void HazeRemove::getAtmosphericLight(Mat& srcImage, double topPercent)
{
	int hist[256];
	double icount = 1.0 / (double)dark.size().area();
	for (int i = 0; i < 256; i++)hist[i] = 0;
	for (int j = 0; j < srcImage.rows; j++)
	{
		uchar* s = dark.ptr(j);
		for (int i = 0; i < srcImage.cols; i++)
		{
			hist[s[i]]++;
		}
	}

	int thresh = 0;
	int v = 0;
	for (int i = 255; i >= 0; i--)
	{
		v += hist[i];
		if (100.0*v*icount > topPercent)
		{
			thresh = i;
			break;
		}
	}
	Mat map;
	threshold(dark, map, thresh, 255, THRESH_BINARY);
	A = CV_RGB(0, 0, 0);

	int maxv = 0;
	int count = 0;
	for (int j = 0; j < srcImage.rows; j++)
	{
		uchar* m = map.ptr(j);
		uchar* s = srcImage.ptr(j);
		for (int i = 0; i < srcImage.cols; i++)
		{
			if (m[i] == 255)
			{
				if (maxv < s[3 * i] + s[3 * i + 1] + s[3 * i + 2])
				{
					A.val[0] += s[3 * i];
					A.val[1] += s[3 * i + 1];
					A.val[2] += s[3 * i + 2];
					count++;
				}
			}

		}
	}
	if (count != 0)
	{
		A.val[0] /= (double)count;
		A.val[1] /= (double)count;
		A.val[2] /= (double)count;
	}
}

void HazeRemove::getTransmissionMap(float omega)
{
	tmap = Mat::ones(size, CV_32F);
	const float ir = (float)(1.0 / A.val[0]);
	const float ig = (float)(1.0 / A.val[1]);
	const float ib = (float)(1.0 / A.val[2]);

	//#pragma omp parallel for
	for (int i = 0; i < size.height; i++)
	{
		uchar* sr = minvalue[0].ptr<uchar>(i);
		uchar* sg = minvalue[1].ptr<uchar>(i);
		uchar* sb = minvalue[2].ptr<uchar>(i);
		float* dptr = tmap.ptr<float>(i);

		for (int j = 0; j < size.width; j++)
		{
			float minv = min((float)sr[j] * ir, (float)sg[j] * ig);
			dptr[j] -= omega*min(minv, (float)sb[j] * ib);
		}
	}
}

void HazeRemove::removeHaze(Mat& src, Mat& trans, Scalar v, Mat& dest, float clip)
{
	if (dest.empty())dest = Mat::zeros(src.size(), src.type());

	for (int j = 0; j < src.rows; j++)
	{
		float* a = trans.ptr<float>(j);
		uchar* s = src.ptr(j);
		uchar* d = dest.ptr(j);
		for (int i = 0; i < src.cols; i++)
		{
			float t = max(clip, a[i]);

			d[3 * i + 0] = saturate_cast<uchar>((s[3 * i + 0] - v.val[0]) / t + v.val[0]);
			d[3 * i + 1] = saturate_cast<uchar>((s[3 * i + 1] - v.val[1]) / t + v.val[1]);
			d[3 * i + 2] = saturate_cast<uchar>((s[3 * i + 2] - v.val[2]) / t + v.val[2]);
		}
	}
}

HazeRemove::HazeRemove()
{
	minvalue.resize(3);
	filter_switch=0;
	sigma_range = 25.f;
}

HazeRemove::~HazeRemove()
{
	;
}

void HazeRemove::getAtmosphericLightImage(Mat& dest)
{
	if (dest.empty())dest.create(size, CV_8UC3);
	dest.setTo(A);
}

void HazeRemove::showTransmissionMap(Mat& dest, bool isPseudoColor)
{
	Mat temp;
	tmap.convertTo(temp, CV_8U, 255);
	if (!isPseudoColor)cvtColor(temp, dest, COLOR_GRAY2BGR);
	else applyColorMap(temp, dest, 2);
}

void HazeRemove::showDarkChannel(Mat& dest, bool isPseudoColor)
{
	if (!isPseudoColor)cvtColor(dark, dest, COLOR_GRAY2BGR);
	else applyColorMap(dark, dest, 2);
}

void HazeRemove::operator() (Mat& src, Mat& dest, int r_dark, double toprate, int r_joint, double e_joint)
{
	size = src.size();

	darkChannel(src, r_dark);
	getAtmosphericLight(src, toprate);
	getTransmissionMap();
	Mat srcg;
	cvtColor(src, srcg, COLOR_BGR2GRAY);

	ArbitraryWindowedGuidedImageFilter gf;
	gf.setMode(filter_switch);
	gf.setSigmaRange(sigma_range);
	gf.filter(tmap, srcg, tmap, r_joint, (float)e_joint);
	removeHaze(src, tmap, A, dest);
}

void HazeRemove::run(Mat& src, string wname)
{
	namedWindow(wname);
	filter_switch = 0; createTrackbar("swich", wname, &filter_switch, 3);
	int ksize = 15;
	createTrackbar("r", wname, &ksize, 300);
	int eps = 60;
	createTrackbar("eps*10", wname, &eps, 255);
	int sigma_range = 25;
	createTrackbar("sigma_range", wname, &sigma_range, 255);

	int hazerate = 10;
	createTrackbar("hazerate", wname, &hazerate, 100);
	int hazesize = 4;
	createTrackbar("hazesize", wname, &hazesize, 100);
	
	int key = 0;
	while (key != 'q')
	{
		Mat show;
		Mat destC;
		Mat destDark;
		{
			//CalcTime t("dehaze");
			float e = (eps*0.1) * (eps*0.1);
			operator()(src, show, hazesize, hazerate / 100.0, ksize, e);

		}
		showTransmissionMap(destC, false);
		Mat a;
		getAtmosphericLightImage(a);
		imshow("a light", a);

		vector<Mat> v;
		v.push_back(show);
		v.push_back(Mat(Scalar::all(255) - destC));
		v.push_back(src);

		imshowAnalysis(wname, v);
		//imshow(wname, destC);

		key = waitKey(33);
	}
	destroyAllWindows();
}