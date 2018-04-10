#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "util.hpp"
#include "arbitraryWindowedGuidedImageFilter.hpp"
#include "hazeRemove.hpp"

using namespace cv;
using namespace std;

#define CV_LIB_PREFIX comment(lib, "opencv_"
#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib")
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib")
#endif
#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
CVAUX_STR(CV_MINOR_VERSION)\
CVAUX_STR(CV_SUBMINOR_VERSION)
#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX
#pragma CV_LIBRARY(core)
#pragma CV_LIBRARY(imgcodecs)
#pragma CV_LIBRARY(highgui)
#pragma CV_LIBRARY(imgproc)


void ArbitraryWindowedGuidedImageFilterTest(Mat& src, bool isShowProfilePlot = true, String wname = "AWGIF")
{
	const int width = src.cols;
	const int height = src.rows;
	namedWindow(wname);
	ConsoleImage ci;

	//for synthetic signal
	int xbox1 = width / 2;  createTrackbar("xbox1", "", &xbox1, width);
	int ybox1 = height / 2; createTrackbar("ybox1", "", &ybox1, height);
	int rbox1 = 100; createTrackbar("rbox1", "", &rbox1, 200);
	int obox1 = 20; createTrackbar("offset box1", "", &obox1, 128);
	int bbox1 = 0; createTrackbar("blur box1", "", &bbox1, 100);
	int xbox2 = width / 2;  createTrackbar("xbox2", "", &xbox2, width);
	int ybox2 = height / 2; createTrackbar("ybox2", "", &ybox2, height);
	int rbox2 = 100; createTrackbar("rbox2", "", &rbox2, 200);
	int oboxt2 = 50; createTrackbar("offset box top", "", &oboxt2, 128);
	int obox2 = 50; createTrackbar("offset box2", "", &obox2, 128);
	int bbox2 = 5; createTrackbar("blur box2", "", &bbox2, 100);

	int sw = 1; createTrackbar("swich", wname, &sw, 3);
	int r = 2; createTrackbar("r", wname, &r, 100);
	int eps = 10; createTrackbar("eps*10", wname, &eps, 1000);
	int sigma_r = 25; createTrackbar("sigma_range", wname, &sigma_r, 500);
	int iteration = 1; createTrackbar("iteration", wname, &iteration, 10);

	int noise = 0; createTrackbar("noise sigma", wname, &noise, 100);
	int amp = 20; createTrackbar("amp detail enhance", wname, &amp, 100);

	int a = 0; createTrackbar("alpha", wname, &a, 100);
	
	int key = 0;
	Mat guidesig = src.clone();
	Mat srcsig = src.clone();
	Mat show;

	ArbitraryWindowedGuidedImageFilter awgif;
	Mat input;
	Stat stat;
	bool isUpdate = true;

	while (key != 'q')
	{		
		/*
		//generate synthetic signal
		guidesig.setTo(obox1);
		srcsig.setTo(obox2);
		rectangle(guidesig, Rect(xbox1 - rbox1, ybox1 - rbox1, 2 * rbox1 + 1, 2 * rbox1 + 1), cvScalarAll(255-obox1), CV_FILLED);
		blur(guidesig, guidesig, Size(2 * bbox1 + 1, 2 * bbox1 + 1));
		rectangle(srcsig, Rect(xbox2 - rbox2, ybox2 - rbox2, 2 * rbox2 + 1, 2 * rbox2 + 1), cvScalarAll(255-oboxt2), CV_FILLED);
		GaussianFilter(srcsig, srcsig, bbox2*0.333, GAUSSIAN_FILTER_DERICHE);
		//blur(srcsig, srcsig, Size(2 * bbox2 + 1, 2 * bbox2 + 1));
		//addNoise(guidesig, guidesig, 10);
		*/

		Mat ideal = srcsig.clone();
		if (isUpdate)
			addNoise(srcsig, input, noise);
		if (key == 'u')isUpdate = (isUpdate) ? false : true;

		float e = (float)((eps/10.0) * (eps/10.0));

		Mat showf;
		input.convertTo(showf, CV_32F);

		CalcTime t("time", 0, false);
		awgif.setMode(sw);
		awgif.setSigmaRange((float)sigma_r);
		for (int i = 0; i < iteration; i++)
			awgif.filter(showf, showf, showf, r, e);
		stat.push_back(t.getTime());

		if(sw==0) ci("Box");
		else if (sw == 1) ci("Dual Exponential");
		if (sw == 2) ci("Gaussian");
		if (sw == 3) ci("Bilateral");
		ci(format("time %f [ms]", stat.getMedian()));
		
		showf.convertTo(show, CV_8U);
		ci(format("PSNR all  %f", calcPSNR(ideal, show)));
		ci.flush();

		alphaBlend(show, guidesig, 1.0 - a / 100.0, show);
		vector<Mat> ss;
		ss.push_back(show);
		ss.push_back(ideal);

		Mat detail = srcsig + amp*0.1*(srcsig - show);
		imshow("detail", detail);
		ss.push_back(detail);

		//Mat diff = amp*0.1*(show- ideal)+128.0;
		Mat diff = amp*0.1*(show - srcsig) + 128.0;
		ss.push_back(diff);

		if(isShowProfilePlot) imshowAnalysis(wname, ss);
		else imshow(wname, show);
		

		key = waitKey(1);
		if (key == 'r') stat.clear();
		if (key == 'n') isUpdate = (isUpdate) ? false : true;
	}
	destroyAllWindows();
}

class VisualizeKernel
{
public:
	string wname;
	static void onMouse(int event, int x, int y, int flags, void* param)
	{
		Point* ret = (Point*)param;

		if (flags == CV_EVENT_FLAG_LBUTTON)
		{
			ret->x = x;
			ret->y = y;
		}
	}

	virtual void filter(Mat& src, Mat& guide, Mat& dest) = 0;
	virtual void setUpTrackbar() = 0;

	void showProfile(Mat& src, Point pt)
	{
		Mat show(Size(src.cols, 255), CV_8U);
		show.setTo(255);
		for (int i = 0; i < src.cols - 1; i++)
		{
			line(show, Point(i, src.at<float>(pt.y, i)),
				Point(i + 1, src.at<float>(pt.y, i + 1)), COLOR_BLACK);
		}
		flip(show, show, 0);
		imshow("plofile", show);
	}

	void run(Mat& src, const int maxKenelPlots = 1, Point pt = Point(0, 0), string winname = "viz")
	{
		wname = winname;
		namedWindow(wname);
		int ptindex = 0;
		setUpTrackbar();
		if (maxKenelPlots >= 2)
		{
			createTrackbar("pt_index", wname, &ptindex, maxKenelPlots);
		}
		int a = 20; createTrackbar("alpha", wname, &a, 100);
		int base = 64; createTrackbar("base", wname, &base, 128);
		int noise = 0; createTrackbar("noise", wname, &noise, 100);
		
		vector<Point> pts(maxKenelPlots);
		if (pt.x == 0 && pt.y == 0)
		{
			pt = Point(src.cols / 2, src.rows / 2);
			for (int i = 0; i < maxKenelPlots; i++)
			{
				pts[i] = Point(src.cols / 2, src.rows / 2);
			}
		}
		cv::setMouseCallback(wname, (MouseCallback)onMouse, (void*)&pt);

		Mat srcf; src.convertTo(srcf, CV_32F);
		Mat srcc; cvtColor(src, srcc, COLOR_GRAY2BGR);
		Mat dest;
		Mat show;
		Mat point = Mat::ones(src.size(), CV_32F);

		int key = 0;
		while (key != 'q')
		{
			point.setTo(FLT_EPSILON);
			for (int i = 0; i < maxKenelPlots; i++)
			{
				point.at <float>(pts[i]) = 25500.0;
			}
			pts[ptindex] = pt;

			Mat nf;
			addNoise(srcf, nf, noise);
			filter(nf, nf, dest);
			dest.convertTo(show, CV_8U);
			imshow("filtered", show);
			filter(point, nf, dest);

			double minv, maxv;
			minMaxLoc(dest, &minv, &maxv);
			dest = (255.0 / maxv * dest)*((255.-base)/ 255.0) +base;

			//normalize(dest, dest, 255, 0, NORM_MINMAX);

			showProfile(dest, pt);
			dest.convertTo(show, CV_8U);

			applyColorMap(show, show, 2);
			alphaBlend(srcc, show, a*0.01, show);

			imshow(wname, show);
			key = waitKey(1);
		}
		destroyAllWindows();
	}
};

class VizKernelAWGIF : public VisualizeKernel
{
public:

	int fa;
	int r;
	int iter;
	int eps;
	int sigmaR;

	int sw;
	void setUpTrackbar()
	{
		sw = 0; createTrackbar("swich", wname, &sw, 3);
		r = 10; createTrackbar("r", wname, &r, 50);
		eps = 10; createTrackbar("eps*10", wname, &eps, 1000);
		sigmaR = 25; createTrackbar("sigma_range", wname, &sigmaR, 100);
		iter = 1; createTrackbar("iteration", wname, &iter, 100);
	}

	void filter(Mat& src, Mat& guide, Mat& dest)
	{
		//CalcTime t;
		src.copyTo(dest);
		ArbitraryWindowedGuidedImageFilter gf;
		gf.setMode(sw);
		gf.setSigmaRange(sigmaR);

		float e = (eps/10.0) * (eps/10.0);
		Mat joint = guide.clone();
		for (int i = 0; i < iter; i++)
		{
			gf.filter(dest, joint, dest, r, e);
		}
	}
};

int main(int argc, char** argv)
{
	//for denoising, detail enhancement
	Mat src_ = imread("nagoya.png", 0);
	Mat src;
	const int width = 512*4;
	const int height = 512*4;
	resize(src_, src, Size(width, height));
	ArbitraryWindowedGuidedImageFilterTest(src, false);

	//for kernel visuarization
	Mat lenna = imread("lenna.png", 0);
	VizKernelAWGIF viz;
	viz.run(lenna);

	//for haze removing
	HazeRemove hz;
	Mat hazy = imread("canyon.png");
	hz.run(hazy);

	return 0;
}