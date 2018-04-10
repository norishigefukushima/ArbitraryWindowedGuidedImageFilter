#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "dualExponentialSmoothing.hpp"
#include "GaussianFilterSpectralRecursive.hpp"
#include "jointBilateralFilter.hpp"

using namespace cv;
using namespace std;

class ArbitraryWindowedGuidedImageFilter
{
private:
	int mode;
	int BORDER_TYPE;
	float sigma_range;
	LaplacianSmoothingIIR LaplacianIIR;
	void internal_filter(Mat& src, Mat& dest, int r, Mat& guide=Mat())
	{
		int tap = 4;
		float sigma = r / 3.f;
		if (mode == 0)
		{
			boxFilter(src, dest, src.depth(), Size(2 * r + 1, 2 * r + 1), Point(-1, -1), true);
		}
		else if (mode == 1)
		{
			double b = 1.0 - exp(-1 / (sigma));
			//LaplacianSmoothingIIRFilter(src, dest, b);
			LaplacianIIR.filter(src, dest, b);
		}
		else if (mode == 2)
		{
			GaussianBlurSR(src, dest, sigma);
		}
		else if (mode == 3)
		{
			int r2 = cvRound(1.4142*r);
			jointBilateralFilter(src, guide, dest, r2, sigma_range, sigma, FILTER_CIRCLE, BORDER_REPLICATE);
		}
	}

public:
	ArbitraryWindowedGuidedImageFilter()
	{
		int BORDER_TYPE = BORDER_REPLICATE;
		mode = 0;
		sigma_range = 25.f;
	}

	void setMode(const int m)
	{
		mode = m;
	}

	void setSigmaRange(const float sigma_range_)
	{
		sigma_range = sigma_range_;
	}

	void filter(const Mat& src, const Mat& joint, Mat& dest, const int radius, const float eps)
	{
		int BORDER_TYPE = BORDER_REPLICATE;
		Size ksize(2 * radius + 1, 2 * radius + 1);
		Size imsize = src.size();
		const float e = eps;

		Mat sf; src.convertTo(sf, CV_32F);
		Mat jf; joint.convertTo(jf, CV_32F);

		Mat mJoint(imsize, CV_32F);//mean_I
		Mat mSrc(imsize, CV_32F);//mean_p

		internal_filter(jf, mJoint, radius, jf);//mJoint*K/////////////////////////
		internal_filter(sf, mSrc, radius, jf);//mSrc*K

		Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

		multiply(jf, sf, x1);//x1*1
		internal_filter(x1, x3, radius , jf);//corrI
		multiply(mJoint, mSrc, x1);//;x1*K*K
		x3 -= x1;//x1 div k ->x3*k
		multiply(jf, jf, x1);////////////////////////////////////
		internal_filter(x1, x2, radius, jf);//x2*K
		multiply(mJoint, mJoint, x1);//x1*K*K

		sf = Mat(x2 - x1) + e;
		divide(x3, sf, x3);//x3->a
		multiply(x3, mJoint, x1);
		x1 -= mSrc;//x1->b
		internal_filter(x3, x2, radius, jf);//x2*k
		internal_filter(x1, x3, radius, jf);//x3*k
		multiply(x2, jf, x1);//x1*K
		Mat temp = x1 - x3;//

		if (src.depth() == CV_8U)
			temp.convertTo(dest, src.type(), 1, 0.5);
		else if(src.depth() ==CV_32F) temp.copyTo(dest);
		else 	temp.convertTo(dest, src.type(), 1);
	}
};