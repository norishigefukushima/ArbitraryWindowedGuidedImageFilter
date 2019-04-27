#include "util.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <stdarg.h>
#include <iostream>
using namespace cv;
using namespace std;

void triangle(InputOutputArray src_, Point pt, int length, Scalar& color, int thickness)
{
	Mat src = src_.getMat();
	int npt[] = { 3, 0 };
	cv::Point pt1[1][3];
	const int h = cvRound(1.7320508*0.5*length);
	pt1[0][0] = Point(pt.x, pt.y - h / 2);;
	pt1[0][1] = Point(pt.x + length / 2, pt.y + h / 2);
	pt1[0][2] = Point(pt.x - length / 2, pt.y + h / 2);

	const cv::Point *ppt[1] = { pt1[0] };
	
	if (thickness == cv::FILLED)
	{
		fillPoly(src, ppt, npt, 1, color, 1);
	}
	else
	{
		polylines(src, ppt, npt, 1, true, color, thickness);
	}
	src.copyTo(src_);
}

void triangleinv(InputOutputArray src_, Point pt, int length, Scalar& color, int thickness)
{
	Mat src = src_.getMat();
	int npt[] = { 3, 0 };
	cv::Point pt1[1][3];
	const int h = cvRound(1.7320508*0.5*length);
	pt1[0][0] = Point(pt.x, pt.y + h / 2);;
	pt1[0][1] = Point(pt.x + length / 2, pt.y - h / 2);
	pt1[0][2] = Point(pt.x - length / 2, pt.y - h / 2);

	const cv::Point *ppt[1] = { pt1[0] };

	if (thickness == cv::FILLED)
	{
		fillPoly(src, ppt, npt, 1, color, 1);
	}
	else
	{
		polylines(src, ppt, npt, 1, true, color, thickness);
	}
	src.copyTo(src_);
}

void drawPlus(InputOutputArray src, Point crossCenter, int length, Scalar& color, int thickness, int line_type, int shift)
{
	Mat dest = src.getMat();
	if (crossCenter.x == 0 && crossCenter.y == 0)
	{
		crossCenter.x = dest.cols / 2;
		crossCenter.y = dest.rows / 2;
	}

	int hl = length / 2;
	line(dest, cv::Point(crossCenter.x - hl, crossCenter.y), cv::Point(crossCenter.x + hl, crossCenter.y), color, thickness, line_type, shift);
	line(dest, cv::Point(crossCenter.x, crossCenter.y - hl), cv::Point(crossCenter.x, crossCenter.y + hl), color, thickness, line_type, shift);

	dest.copyTo(src);
}

void drawTimes(InputOutputArray src, Point crossCenter, int length, Scalar& color, int thickness, int line_type, int shift)
{
	Mat dest = src.getMat();
	if (crossCenter.x == 0 && crossCenter.y == 0)
	{
		crossCenter.x = dest.cols / 2;
		crossCenter.y = dest.rows / 2;
	}
	int hl = cvRound((double)length / 2.0 / sqrt(2.0));
	line(dest, cv::Point(crossCenter.x - hl, crossCenter.y - hl), cv::Point(crossCenter.x + hl, crossCenter.y + hl), color, thickness, line_type, shift);
	line(dest, cv::Point(crossCenter.x + hl, crossCenter.y - hl), cv::Point(crossCenter.x - hl, crossCenter.y + hl), color, thickness, line_type, shift);

	dest.copyTo(src);
}

void drawAsterisk(InputOutputArray src, Point crossCenter, int length, Scalar& color, int thickness, int line_type, int shift)
{
	Mat dest = src.getMat();
	if (crossCenter.x == 0 && crossCenter.y == 0)
	{
		crossCenter.x = dest.cols / 2;
		crossCenter.y = dest.rows / 2;
	}

	int hl = cvRound((double)length / 2.0 / sqrt(2.0));
	line(dest, cv::Point(crossCenter.x - hl, crossCenter.y - hl), cv::Point(crossCenter.x + hl, crossCenter.y + hl), color, thickness, line_type, shift);
	line(dest, cv::Point(crossCenter.x + hl, crossCenter.y - hl), cv::Point(crossCenter.x - hl, crossCenter.y + hl), color, thickness, line_type, shift);

	hl = length / 2;
	line(dest, cv::Point(crossCenter.x - hl, crossCenter.y), cv::Point(crossCenter.x + hl, crossCenter.y), color, thickness, line_type, shift);
	line(dest, cv::Point(crossCenter.x, crossCenter.y - hl), cv::Point(crossCenter.x, crossCenter.y + hl), color, thickness, line_type, shift);

	dest.copyTo(src);
}

void CalcTime::start()
{
	pre = getTickCount();
}

void CalcTime::restart()
{
	start();
}

void CalcTime::lap(string message)
{
	string v = message + format(" %f", getTime());
	switch (timeMode)
	{
	case TIME_NSEC:
		v += " NSEC";
		break;
	case TIME_SEC:
		v += " SEC";
		break;
	case TIME_MIN:
		v += " MIN";
		break;
	case TIME_HOUR:
		v += " HOUR";
		break;

	case TIME_MSEC:
	default:
		v += " msec";
		break;
	}
	lap_mes.push_back(v);
	restart();
}

void CalcTime::show()
{
	getTime();

	int mode = timeMode;
	if (timeMode == TIME_AUTO)
	{
		mode = autoMode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		cout << mes << ": " << cTime << " nsec" << endl;
		break;
	case TIME_SEC:
		cout << mes << ": " << cTime << " sec" << endl;
		break;
	case TIME_MIN:
		cout << mes << ": " << cTime << " minute" << endl;
		break;
	case TIME_HOUR:
		cout << mes << ": " << cTime << " hour" << endl;
		break;

	case TIME_MSEC:
	default:
		cout << mes << ": " << cTime << " msec" << endl;
		break;
	}
}

void CalcTime::show(string mes)
{
	getTime();

	int mode = timeMode;
	if (timeMode == TIME_AUTO)
	{
		mode = autoMode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		cout << mes << ": " << cTime << " nsec" << endl;
		break;
	case TIME_SEC:
		cout << mes << ": " << cTime << " sec" << endl;
		break;
	case TIME_MIN:
		cout << mes << ": " << cTime << " minute" << endl;
		break;
	case TIME_HOUR:
		cout << mes << ": " << cTime << " hour" << endl;
		break;
	case TIME_DAY:
		cout << mes << ": " << cTime << " day" << endl;
	case TIME_MSEC:
		cout << mes << ": " << cTime << " msec" << endl;
		break;
	default:
		cout << mes << ": error" << endl;
		break;
	}
}

int CalcTime::autoTimeMode()
{
	if (cTime > 60.0*60.0*24.0)
	{
		return TIME_DAY;
	}
	else if (cTime > 60.0*60.0)
	{
		return TIME_HOUR;
	}
	else if (cTime > 60.0)
	{
		return TIME_MIN;
	}
	else if (cTime > 1.0)
	{
		return TIME_SEC;
	}
	else if (cTime > 1.0 / 1000.0)
	{
		return TIME_MSEC;
	}
	else
	{

		return TIME_NSEC;
	}
}

double CalcTime::getTime()
{
	cTime = (getTickCount() - pre) / (getTickFrequency());

	int mode = timeMode;
	if (mode == TIME_AUTO)
	{
		mode = autoTimeMode();
		autoMode = mode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		cTime *= 1000000.0;
		break;
	case TIME_SEC:
		cTime *= 1.0;
		break;
	case TIME_MIN:
		cTime /= (60.0);
		break;
	case TIME_HOUR:
		cTime /= (60 * 60);
		break;
	case TIME_DAY:
		cTime /= (60 * 60 * 24);
		break;
	case TIME_MSEC:
	default:
		cTime *= 1000.0;
		break;
	}
	return cTime;
}

void CalcTime::setMessage(string& src)
{
	mes = src;
}

void CalcTime::setMode(int mode)
{
	timeMode = mode;
}

void CalcTime::init(string message, int mode, bool isShow)
{
	_isShow = isShow;
	timeMode = mode;

	setMessage(message);
	start();
}

CalcTime::CalcTime()
{
	string t = "time ";
	init(t, TIME_AUTO, true);
}

CalcTime::CalcTime(char* message, int mode, bool isShow)
{
	string m = message;
	init(m, mode, isShow);
}

CalcTime::CalcTime(string message, int mode, bool isShow)
{
	init(message, mode, isShow);
}

CalcTime::~CalcTime()
{
	getTime();
	if (_isShow)	show();
	if (lap_mes.size() != 0)
	{
		for (int i = 0; i < lap_mes.size(); i++)
		{
			cout << lap_mes[i] << endl;
		}
	}
}

void alphaBlend(InputArray src1, InputArray src2, const double alpha, OutputArray dest)
{
	CV_Assert(src1.size() == src2.size());
	Mat s1, s2;
	if (src1.depth() == src2.depth())
	{
		if (src1.channels() == src2.channels())
		{
			s1 = src1.getMat();
			s2 = src2.getMat();
		}
		else if (src2.channels() == 3)
		{
			cvtColor(src1, s1, COLOR_GRAY2BGR);
			s2 = src2.getMat();
		}
		else
		{
			cvtColor(src2, s2, COLOR_GRAY2BGR);
			s1 = src1.getMat();
		}
	}
	else if (src1.depth() != src2.depth())
	{
		int depth = max(src1.depth(), src2.depth());

		if (src1.channels() == src2.channels())
		{
			s1 = src1.getMat();
			s2 = src2.getMat();
		}
		else if (src2.channels() == 3)
		{
			cvtColor(src1, s1, COLOR_GRAY2BGR);
			s2 = src2.getMat();
		}
		else
		{
			cvtColor(src2, s2, COLOR_GRAY2BGR);
			s1 = src1.getMat();
		}
		s1.convertTo(s1, depth);
		s1.convertTo(s2, depth);
	}

	cv::addWeighted(s1, alpha, s2, 1.0 - alpha, 0.0, dest);
}

double calcPSNR(InputArray I1_, InputArray I2_)
{

	Mat I1, I2;
	if (I1_.channels() == 1 && I2_.channels() == 1)
	{
		I1_.getMat().convertTo(I1, CV_64F);
		I2_.getMat().convertTo(I2, CV_64F);
	}
	if (I1_.channels() == 3 && I2_.channels() == 3)
	{
		Mat temp;
		cvtColor(I1_, temp, COLOR_BGR2GRAY);
		temp.convertTo(I1, CV_64F);
		cvtColor(I2_, temp, COLOR_BGR2GRAY);
		temp.convertTo(I2, CV_64F);
	}

	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);        // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((255.0 * 255.0) / mse);
		return psnr;
	}
}

template <class T>
void addNoiseSoltPepperMono_(Mat& src, Mat& dest, double per)
{
	cv::RNG rng;
	for (int j = 0; j < src.rows; j++)
	{
		T* s = src.ptr<T>(j);
		T* d = dest.ptr<T>(j);
		for (int i = 0; i < src.cols; i++)
		{
			double a1 = rng.uniform((double)0, (double)1);

			if (a1 > per)
				d[i] = s[i];
			else
			{
				double a2 = rng.uniform((double)0, (double)1);
				if (a2 > 0.5)d[i] = (T)0.0;
				else d[i] = (T)255.0;
			}
		}
	}
}

void addNoiseSoltPepperMono(Mat& src, Mat& dest, double per)
{
	if (src.type() == CV_8U) addNoiseSoltPepperMono_<uchar>(src, dest, per);
	if (src.type() == CV_16U) addNoiseSoltPepperMono_<ushort>(src, dest, per);
	if (src.type() == CV_16S) addNoiseSoltPepperMono_<short>(src, dest, per);
	if (src.type() == CV_32S) addNoiseSoltPepperMono_<int>(src, dest, per);
	if (src.type() == CV_32F) addNoiseSoltPepperMono_<float>(src, dest, per);
	if (src.type() == CV_64F) addNoiseSoltPepperMono_<double>(src, dest, per);
}

void addNoiseMono_nf(Mat& src, Mat& dest, double sigma)
{
	Mat s;
	src.convertTo(s, CV_32S);
	Mat n(s.size(), CV_32S);
	randn(n, 0, sigma);
	Mat temp = s + n;
	temp.convertTo(dest, src.type());
}

void addNoiseMono_f(Mat& src, Mat& dest, double sigma)
{
	Mat s;
	src.convertTo(s, CV_64F);
	Mat n(s.size(), CV_64F);
	randn(n, 0, sigma);
	Mat temp = s + n;
	temp.convertTo(dest, src.type());
}

void addNoiseMono(Mat& src, Mat& dest, double sigma)
{
	if (src.type() == CV_32F || src.type() == CV_64F)
	{
		addNoiseMono_f(src, dest, sigma);
	}
	else
	{
		addNoiseMono_nf(src, dest, sigma);
	}
}

void addNoise(InputArray src_, OutputArray dest_, double sigma, double sprate)
{
	if (dest_.empty() || dest_.size() != src_.size() || dest_.type() != src_.type()) dest_.create(src_.size(), src_.type());
	Mat src = src_.getMat();
	Mat dest = dest_.getMat();
	if (src.channels() == 1)
	{
		addNoiseMono(src, dest, sigma);
		if (sprate != 0)addNoiseSoltPepperMono(dest, dest, sprate);
		return;
	}
	else
	{
		vector<Mat> s(src.channels());
		vector<Mat> d(src.channels());
		split(src, s);
		for (int i = 0; i < src.channels(); i++)
		{
			addNoiseMono(s[i], d[i], sigma);
			if (sprate != 0)addNoiseSoltPepperMono(d[i], d[i], sprate);
		}
		cv::merge(d, dest);
	}
}

double calcPSNR(Mat& src1, Mat& src2, Rect rect)
{
	Mat s1 = src1(rect);
	Mat s2 = src2(rect);
	return calcPSNR(s1, s2);
}

void ConsoleImage::init(Size size, string wname)
{
	isLineNumber = false;
	windowName = wname;
	show = Mat::zeros(size, CV_8UC3);
	clear();
}
ConsoleImage::ConsoleImage()
{
	init(Size(640, 480), "console");
}
ConsoleImage::ConsoleImage(Size size, string wname)
{
	init(size, wname);
}
ConsoleImage::~ConsoleImage()
{
	printData();
}
void ConsoleImage::setIsLineNumber(bool isLine)
{
	isLineNumber = isLine;
}

bool ConsoleImage::getIsLineNumber()
{
	return isLineNumber;
}
void ConsoleImage::printData()
{
	for (int i = 0; i < (int)strings.size(); i++)
	{
		cout << strings[i] << endl;
	}
}
void ConsoleImage::clear()
{
	count = 0;
	show.setTo(0);
	strings.clear();
}
void ConsoleImage::flush(bool isClear)
{
	imshow(windowName, show);
	if (isClear)clear();
}
void ConsoleImage::operator()(string src)
{
	if (isLineNumber)strings.push_back(format("%2d ", count) + src);
	else strings.push_back(src);

	cv::putText(show, src, Point(20, 20 + count * 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255), 1);
	count++;
}
void ConsoleImage::operator()(const char *format, ...)
{
	char buff[255];

	va_list ap;
	va_start(ap, format);
	vsprintf_s(buff, format, ap);
	va_end(ap);

	string a = buff;

	if (isLineNumber)strings.push_back(cv::format("%2d ", count) + a);
	else strings.push_back(a);

	cv::putText(show, buff, Point(20, 20 + count * 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255), 1);
	count++;
}

void ConsoleImage::operator()(cv::Scalar color, const char *format, ...)
{
	char buff[255];

	va_list ap;
	va_start(ap, format);
	vsprintf_s(buff, format, ap);
	va_end(ap);

	string a = buff;
	if (isLineNumber)strings.push_back(cv::format("%2d ", count) + a);
	else strings.push_back(a);
	cv::putText(show, buff, Point(20, 20 + count * 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255), 1);
	count++;
}

Stat::Stat() { ; }
Stat::~Stat() { ; }

double Stat::getMin()
{
	double minv = DBL_MAX;
	for (int i = 0; i < num_data; i++)
	{
		minv = min(minv, data[i]);
	}
	return minv;
}

double Stat::getMax()
{
	double maxv = DBL_MIN;
	for (int i = 0; i < num_data; i++)
	{
		maxv = max(maxv, data[i]);
	}
	return maxv;
}

double Stat::getMean()
{
	double sum = 0.0;
	for (int i = 0; i < num_data; i++)
	{
		sum += data[i];
	}
	return sum / (double)num_data;
}

double Stat::getStd()
{
	double std = 0.0;
	double mean = getMean();
	for (int i = 0; i < num_data; i++)
	{
		std += (mean - data[i])*(mean - data[i]);
	}
	return sqrt(std / (double)num_data);
}

double Stat::getMedian()
{
	if (data.size() == 0) return 0.0;
	vector<double> v;
	vector<double> s;
	for (int i = 0; i < data.size(); i++)
	{
		s.push_back(data[i]);
	}
	cv::sort(s, v, cv::SORT_ASCENDING);
	return v[num_data / 2];
}

void Stat::push_back(double val)
{
	data.push_back(val);
	num_data = (int)data.size();
}

void Stat::clear()
{
	data.clear();
	num_data = 0;
}

void Stat::show()
{
	cout << "mean " << getMean() << endl;
	cout << "min  " << getMin() << endl;
	cout << "med  " << getMedian() << endl;
	cout << "max  " << getMax() << endl;
	cout << "std  " << getStd() << endl;
}

void drawGrid(InputOutputArray src, Point crossCenter, Scalar& color, int thickness, int line_type, int shift)
{
	Mat dest = src.getMat();
	if (crossCenter.x == 0 && crossCenter.y == 0)
	{
		crossCenter.x = dest.cols / 2;
		crossCenter.y = dest.rows / 2;
	}

	line(dest, Point(0, crossCenter.y), Point(dest.cols, crossCenter.y), color, thickness, line_type, shift);
	line(dest, Point(crossCenter.x, 0), Point(crossCenter.x, dest.rows), color, thickness, line_type, shift);

	dest.copyTo(src);
}

template <class T>
void getImageLine_(Mat& src, int channel, vector<Point>& v, const int line)
{
	const int ch = src.channels();

	T* s = src.ptr<T>(line);
	if (ch == 1)
	{
		for (int i = 0; i < src.cols; i++)
		{
			v[i] = Point(i, (int)(s[i]));
		}

	}
	else if (ch == 3)
	{
		if (channel < 0 || channel>2)
		{
			for (int i = 0; i < src.cols; i++)
			{
				v[i] = Point(i, (int)(0.299*s[3 * i + 2] + 0.587*s[3 * i + 1] + 0.114*s[3 * i + 0]));
			}
		}
		else
		{
			for (int i = 0; i < src.cols; i++)
			{
				v[i] = Point(i, (int)(s[3 * i + channel]));
			}
		}
	}
}

static void getImageLine(Mat& src, vector<Point>& v, const int line, int channel)
{
	if (v.size() == 0)
		v.resize(src.cols);

	if (src.type() == CV_8U || src.type() == CV_8UC3)getImageLine_<uchar>(src, channel, v, line);
	else if (src.type() == CV_16S || src.type() == CV_16SC3)getImageLine_<short>(src, channel, v, line);
	else if (src.type() == CV_16U || src.type() == CV_16UC3)getImageLine_<ushort>(src, channel, v, line);
	else if (src.type() == CV_32F || src.type() == CV_32FC3)getImageLine_<float>(src, channel, v, line);
	else if (src.type() == CV_64F || src.type() == CV_64FC3)getImageLine_<double>(src, channel, v, line);
}

static void getImageVLine(Mat& src, vector<Point>& v, const int line, int channel)
{
	if (v.size() == 0)
		v.resize(src.rows);

	Mat srct = src.t();

	if (src.type() == CV_8U || src.type() == CV_8UC3)getImageLine_<uchar>(srct, channel, v, line);
	else if (src.type() == CV_16S || src.type() == CV_16SC3)getImageLine_<short>(srct, channel, v, line);
	else if (src.type() == CV_16U || src.type() == CV_16UC3)getImageLine_<ushort>(srct, channel, v, line);
	else if (src.type() == CV_32F || src.type() == CV_32FC3)getImageLine_<float>(srct, channel, v, line);
	else if (src.type() == CV_64F || src.type() == CV_64FC3)getImageLine_<double>(srct, channel, v, line);
}


void plotGraph(OutputArray graph_, vector<Point2d>& data, double xmin, double xmax, double ymin, double ymax,
	Scalar color, int lt, int isLine, int thickness, int pointSize)

{
	CV_Assert(!graph_.empty());
	const int ps = pointSize;

	Mat graph = graph_.getMat();
	double x = (double)graph.cols / (xmax - xmin);
	double y = (double)graph.rows / (ymax - ymin);

	int H = graph.rows - 1;
	const int size = (int)data.size();

	for (int i = 0; i < size; i++)
	{
		double src = data[i].x;
		double dest = data[i].y;

		cv::Point p = Point(cvRound(x*(src - xmin)), H - cvRound(y*(dest - ymin)));

		if (isLine == Plot::LINE_NONE)
		{
			;
		}
		else if (isLine == Plot::LINE_LINEAR)
		{
			if (i != size - 1)
			{
				double nsrc = data[i + 1].x;
				double ndest = data[i + 1].y;
				line(graph, p, Point(cvRound(x*(nsrc - xmin)), H - cvRound(y*(ndest - ymin))),
					color, thickness);
			}
		}
		else if (isLine == Plot::LINE_H2V)
		{
			if (i != size - 1)
			{
				double nsrc = data[i + 1].x;
				double ndest = data[i + 1].x;
				line(graph, p, cv::Point(cvRound(x*(nsrc - xmin)), p.y), color, thickness);
				line(graph, cv::Point(cvRound(x*(nsrc - xmin)), p.y), cv::Point(cvRound(x*(nsrc - xmin)), H - cvRound(y*(ndest - ymin))), color, thickness);
			}
		}
		else if (isLine == Plot::LINE_V2H)
		{
			if (i != size - 1)
			{
				double nsrc = data[i + 1].x;
				double ndest = data[i + 1].x;
				line(graph, p, cv::Point(p.x, H - cvRound(y*(ndest - ymin))), color, thickness);
				line(graph, cv::Point(p.x, H - cvRound(y*(ndest - ymin))), cv::Point(cvRound(x*(nsrc - xmin)), H - cvRound(y*(ndest - ymin))), color, thickness);
			}
		}

		if (lt == Plot::SYMBOL_NOPOINT)
		{
			;
		}
		else if (lt == Plot::SYMBOL_PLUS)
		{
			drawPlus(graph, p, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::SYMBOL_TIMES)
		{
			drawTimes(graph, p, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::SYMBOL_ASTERRISK)
		{
			drawAsterisk(graph, p, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::SYMBOL_CIRCLE)
		{
			circle(graph, p, ps, color, thickness);
		}
		else if (lt == Plot::SYMBOL_RECTANGLE)
		{
			rectangle(graph, cv::Point(p.x - ps, p.y - ps), cv::Point(p.x + ps, p.y + ps),
				color, thickness);
		}
		else if (lt == Plot::SYMBOL_CIRCLE_FILL)
		{
			circle(graph, p, ps, color, cv::FILLED);
		}
		else if (lt == Plot::SYMBOL_RECTANGLE_FILL)
		{
			rectangle(graph, cv::Point(p.x - ps, p.y - ps), cv::Point(p.x + ps, p.y + ps),
				color, cv::FILLED);
		}
		else if (lt == Plot::SYMBOL_TRIANGLE)
		{
			triangle(graph, p, 2 * ps, color, thickness);
		}
		else if (lt == Plot::SYMBOL_TRIANGLE_FILL)
		{
			triangle(graph, p, 2 * ps, color, cv::FILLED);
		}
		else if (lt == Plot::SYMBOL_TRIANGLE_INV)
		{
			triangleinv(graph, p, 2 * ps, color, thickness);
		}
		else if (lt == Plot::SYMBOL_TRIANGLE_INV_FILL)
		{
			triangleinv(graph, p, 2 * ps, color, cv::FILLED);
		}
	}
}


Plot::Plot(Size plotsize_)
{
	data_max = 1;
	xlabel = "x";
	ylabel = "y";
	setBackGoundColor(COLOR_WHITE);

	origin = Point(64, 64);//default
	plotImage = NULL;
	render = NULL;
	setPlotImageSize(plotsize_);

	keyImage.create(Size(256, 256), CV_8UC3);
	keyImage.setTo(background_color);

	setXYMinMax(0, plotsize.width, 0, plotsize.height);
	isPosition = true;
	init();
}

Plot::~Plot()
{
	;
}

void Plot::point2val(cv::Point pt, double* valx, double* valy)
{
	double x = (double)plotImage.cols / (xmax - xmin);
	double y = (double)plotImage.rows / (ymax - ymin);
	int H = plotImage.rows - 1;

	*valx = (pt.x - (origin.x) * 2) / x + xmin;
	*valy = (H - (pt.y - origin.y)) / y + ymin;
}

void Plot::init()
{
	const int DefaultPlotInfoSize = 64;
	pinfo.resize(DefaultPlotInfoSize);
	for (int i = 0; i < pinfo.size(); i++)
	{
		pinfo[i].symbolType = Plot::SYMBOL_PLUS;
		pinfo[i].lineType = Plot::LINE_LINEAR;
		pinfo[i].thickness = 1;

		double v = (double)i / DefaultPlotInfoSize*255.0;
		pinfo[i].color = getPseudoColor(cv::saturate_cast<uchar>(v));

		pinfo[i].keyname = format("data %02d", i);
	}

	pinfo[0].color = COLOR_RED;
	pinfo[0].symbolType = SYMBOL_PLUS;

	pinfo[1].color = COLOR_GREEN;
	pinfo[1].symbolType = SYMBOL_TIMES;

	pinfo[2].color = COLOR_BLUE;
	pinfo[2].symbolType = SYMBOL_ASTERRISK;

	pinfo[3].color = COLOR_MAGENDA;
	pinfo[3].symbolType = SYMBOL_RECTANGLE;

	pinfo[4].color = CV_RGB(0, 0, 128);
	pinfo[4].symbolType = SYMBOL_RECTANGLE_FILL;

	pinfo[5].color = CV_RGB(128, 0, 0);
	pinfo[5].symbolType = SYMBOL_CIRCLE;

	pinfo[6].color = CV_RGB(0, 128, 128);
	pinfo[6].symbolType = SYMBOL_CIRCLE_FILL;

	pinfo[7].color = CV_RGB(0, 0, 0);
	pinfo[7].symbolType = SYMBOL_TRIANGLE;

	pinfo[8].color = CV_RGB(128, 128, 128);
	pinfo[8].symbolType = SYMBOL_TRIANGLE_FILL;

	pinfo[9].color = CV_RGB(0, 128, 64);
	pinfo[9].symbolType = SYMBOL_TRIANGLE_INV;

	pinfo[10].color = CV_RGB(128, 128, 0);
	pinfo[10].symbolType = SYMBOL_TRIANGLE_INV_FILL;

	setPlotProfile(false, true, false);
	graphImage = render;
}
void Plot::setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_, bool isZeroCross_)
{
	isZeroCross = isZeroCross_;
	isXYMAXMIN = isXYMAXMIN_;
	isXYCenter = isXYCenter_;
}

void Plot::setPlotImageSize(Size s)
{
	plotsize = s;
	plotImage.create(s, CV_8UC3);
	render.create(Size(plotsize.width + 4 * origin.x, plotsize.height + 2 * origin.y), CV_8UC3);
}

void Plot::setXYOriginZERO()
{
	recomputeXYMAXMIN(false);
	xmin = 0;
	ymin = 0;
}

void Plot::setYOriginZERO()
{
	recomputeXYMAXMIN(false);
	ymin = 0;
}

void Plot::setXOriginZERO()
{
	recomputeXYMAXMIN(false);
	xmin = 0;
}

void Plot::recomputeXYMAXMIN(bool isCenter, double marginrate)
{
	if (marginrate<0.0 || marginrate>1.0)marginrate = 1.0;
	xmax = -INT_MAX;
	xmin = INT_MAX;
	ymax = -INT_MAX;
	ymin = INT_MAX;
	for (int i = 0; i < data_max; i++)
	{
		for (int j = 0; j < pinfo[i].data.size(); j++)
		{
			double x = pinfo[i].data[j].x;
			double y = pinfo[i].data[j].y;
			xmax = (xmax < x) ? x : xmax;
			xmin = (xmin > x) ? x : xmin;

			ymax = (ymax < y) ? y : ymax;
			ymin = (ymin > y) ? y : ymin;
		}
	}
	xmax_no_margin = xmax;
	xmin_no_margin = xmin;
	ymax_no_margin = ymax;
	ymin_no_margin = ymin;

	double xmargin = (xmax - xmin)*(1.0 - marginrate)*0.5;
	xmax += xmargin;
	xmin -= xmargin;

	double ymargin = (ymax - ymin)*(1.0 - marginrate)*0.5;
	ymax += ymargin;
	ymin -= ymargin;

	if (isCenter)
	{
		double xxx = abs(xmax);
		double yyy = abs(ymax);
		xxx = (xxx < abs(xmin)) ? abs(xmin) : xxx;
		yyy = (yyy < abs(ymin)) ? abs(ymin) : yyy;

		xmax = xxx;
		xmin = -xxx;
		ymax = yyy;
		ymin = -yyy;

		xxx = abs(xmax_no_margin);
		yyy = abs(ymax_no_margin);
		xxx = (xxx < abs(xmin_no_margin)) ? abs(xmin_no_margin) : xxx;
		yyy = (yyy < abs(ymin_no_margin)) ? abs(ymin_no_margin) : yyy;

		xmax_no_margin = xxx;
		xmin_no_margin = -xxx;
		ymax_no_margin = yyy;
		ymin_no_margin = -yyy;
	}
}

void Plot::setXYMinMax(double xmin_, double xmax_, double ymin_, double ymax_)
{
	xmin = xmin_;
	xmax = xmax_;
	ymin = ymin_;
	ymax = ymax_;

	xmax_no_margin = xmax;
	xmin_no_margin = xmin;
	ymax_no_margin = ymax;
	ymin_no_margin = ymin;
}

void Plot::setXMinMax(double xmin_, double xmax_)
{
	recomputeXYMAXMIN(isXYCenter);
	xmin = xmin_;
	xmax = xmax_;
}

void Plot::setYMinMax(double ymin_, double ymax_)
{
	recomputeXYMAXMIN(isXYCenter);
	ymin = ymin_;
	ymax = ymax_;
}

void Plot::setBackGoundColor(Scalar cl)
{
	background_color = cl;
}

void Plot::setPlotThickness(int plotnum, int thickness_)
{
	pinfo[plotnum].thickness = thickness_;
}

void Plot::setPlotColor(int plotnum, Scalar color_)
{
	pinfo[plotnum].color = color_;
}

void Plot::setPlotLineType(int plotnum, int lineType)
{
	pinfo[plotnum].lineType = lineType;
}

void Plot::setPlotSymbol(int plotnum, int symboltype)
{
	pinfo[plotnum].symbolType = symboltype;
}

void Plot::setPlotKeyName(int plotnum, string name)
{
	pinfo[plotnum].keyname = name;
}

void Plot::setPlot(int plotnum, Scalar color, int symboltype, int
	linetype, int thickness)
{
	setPlotColor(plotnum, color);
	setPlotSymbol(plotnum, symboltype);
	setPlotLineType(plotnum, linetype);
	setPlotThickness(plotnum, thickness);
}

void Plot::setPlotSymbolALL(int symboltype)
{
	for (int i = 0; i < pinfo.size(); i++)
	{
		pinfo[i].symbolType = symboltype;
	}
}

void Plot::setPlotLineTypeALL(int linetype)
{
	for (int i = 0; i < pinfo.size(); i++)
	{
		pinfo[i].lineType = linetype;
	}
}

void Plot::push_back(double x, double y, int plotIndex)
{
	data_max = max(data_max, plotIndex + 1);
	pinfo[plotIndex].data.push_back(Point2d(x, y));
}

void Plot::push_back(vector<cv::Point> point, int plotIndex)
{
	data_max = max(data_max, plotIndex + 1);
	for (int i = 0; i < (int)point.size(); i++)
	{
		push_back(point[i].x, point[i].y, plotIndex);
	}
}

void Plot::push_back(vector<cv::Point2d> point, int plotIndex)
{
	data_max = max(data_max, plotIndex + 1);
	for (int i = 0; i < (int)point.size() - 1; i++)
	{
		push_back(point[i].x, point[i].y, plotIndex);
	}
}

void Plot::erase(int sampleIndex, int plotIndex)
{
	pinfo[plotIndex].data.erase(pinfo[plotIndex].data.begin() + sampleIndex);
}

void Plot::insert(Point2d v, int sampleIndex, int plotIndex)
{
	pinfo[plotIndex].data.insert(pinfo[plotIndex].data.begin() + sampleIndex, v);
}

void Plot::insert(Point v, int sampleIndex, int plotIndex)
{
	insert(Point2d((double)v.x, (double)v.y), sampleIndex, plotIndex);
}

void Plot::insert(double x, double y, int sampleIndex, int plotIndex)
{
	insert(Point2d(x, y), sampleIndex, plotIndex);
}

void Plot::clear(int datanum)
{
	if (datanum < 0)
	{
		for (int i = 0; i < data_max; i++)
			pinfo[i].data.clear();

	}
	else
		pinfo[datanum].data.clear();
}

void Plot::swapPlot(int plotIndex1, int plotIndex2)
{
	swap(pinfo[plotIndex1].data, pinfo[plotIndex2].data);
}

void Plot::makeBB(bool isFont)
{
	render.setTo(background_color);
	Mat roi = render(Rect(origin.x * 2, origin.y, plotsize.width, plotsize.height));
	rectangle(plotImage, Point(0, 0), Point(plotImage.cols - 1, plotImage.rows - 1), COLOR_BLACK, 1);
	plotImage.copyTo(roi);

	if (isFont)
	{
		putText(render, xlabel, Point(render.cols / 2, (int)(origin.y*1.85 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);
		putText(render, ylabel, Point(20, (int)(origin.y*0.25 + plotImage.rows / 2)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		string buff;
		//x coordinate
		buff = format("%.2f", xmin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (xmax - xmin)*0.25 + xmin);
		putText(render, buff, Point((int)(origin.x + plotImage.cols*0.25 + 15), (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (xmax - xmin)*0.5 + xmin);
		putText(render, buff, Point((int)(origin.x + plotImage.cols*0.5 + 45), (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (xmax - xmin)*0.75 + xmin);
		putText(render, buff, Point((int)(origin.x + plotImage.cols*0.75 + 35), (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", xmax);
		putText(render, buff, Point(origin.x + plotImage.cols, (int)(origin.y*1.35 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		//y coordinate
		buff = format("%.2f", ymin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.0 + plotImage.rows)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (ymax - ymin)*0.5 + ymin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.0 + plotImage.rows*0.5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (ymax - ymin)*0.25 + ymin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.0 + plotImage.rows*0.75)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", (ymax - ymin)*0.75 + ymin);
		putText(render, buff, Point(origin.x, (int)(origin.y*1.0 + plotImage.rows*0.25)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);

		buff = format("%.2f", ymax);
		putText(render, buff, Point(origin.x, origin.y), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);
	}
}

void Plot::plotPoint(Point2d point, Scalar color_, int thickness_, int linetype)
{
	vector<Point2d> data;

	data.push_back(Point2d(point.x, ymin));
	data.push_back(Point2d(point.x, ymax));
	data.push_back(Point2d(point.x, point.y));
	data.push_back(Point2d(xmax, point.y));
	data.push_back(Point2d(xmin, point.y));

	plotGraph(plotImage, data, xmin, xmax, ymin, ymax, color_, SYMBOL_NOPOINT, linetype, thickness_);
}

void Plot::plotGrid(int level)
{
	if (level > 0)
	{
		plotPoint(Point2d((xmax - xmin) / 2.0 + xmin, (ymax - ymin) / 2.0 + ymin), COLOR_GRAY150, 1);
	}
	if (level > 1)
	{
		plotPoint(Point2d((xmax - xmin)*1.0 / 4.0 + xmin, (ymax - ymin)*1.0 / 4.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*3.0 / 4.0 + xmin, (ymax - ymin)*1.0 / 4.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*1.0 / 4.0 + xmin, (ymax - ymin)*3.0 / 4.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*3.0 / 4.0 + xmin, (ymax - ymin)*3.0 / 4.0 + ymin), COLOR_GRAY200, 1);
	}
	if (level > 2)
	{
		plotPoint(Point2d((xmax - xmin)*1.0 / 8.0 + xmin, (ymax - ymin)*1.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*3.0 / 8.0 + xmin, (ymax - ymin)*1.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*1.0 / 8.0 + xmin, (ymax - ymin)*3.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*3.0 / 8.0 + xmin, (ymax - ymin)*3.0 / 8.0 + ymin), COLOR_GRAY200, 1);

		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*1.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*1.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*3.0 / 8.0 + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*3.0 / 8.0 + ymin), COLOR_GRAY200, 1);

		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*(1.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*(1.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*(3.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0 + 0.5) + xmin, (ymax - ymin)*(3.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);

		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0) + xmin, (ymax - ymin)*(1.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0) + xmin, (ymax - ymin)*(1.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(1.0 / 8.0) + xmin, (ymax - ymin)*(3.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
		plotPoint(Point2d((xmax - xmin)*(3.0 / 8.0) + xmin, (ymax - ymin)*(3.0 / 8.0 + 0.5) + ymin), COLOR_GRAY200, 1);
	}
}

void Plot::makeKey(int num)
{
	int step = 20;
	keyImage.create(Size(256, 20 * (num + 1) + 3), CV_8UC3);
	keyImage.setTo(background_color);

	int height = (int)(0.8*keyImage.rows);
	for (int i = 0; i < num; i++)
	{
		vector<Point2d> data;
		data.push_back(Point2d(192.0, keyImage.rows - (i + 1) * 20));
		data.push_back(Point2d(keyImage.cols - 20, keyImage.rows - (i + 1) * 20));


		plotGraph(keyImage, data, 0, keyImage.cols, 0, keyImage.rows, pinfo[i].color, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].thickness);
		putText(keyImage, pinfo[i].keyname, Point(0, (i + 1) * 20 + 3), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, pinfo[i].color);
	}
}

void Plot::plotData(int gridlevel, int isKey)
{
	plotImage.setTo(background_color);
	plotGrid(gridlevel);

	if (isZeroCross)	plotPoint(Point2d(0.0, 0.0), COLOR_ORANGE, 1);

	for (int i = 0; i < data_max; i++)
	{
		plotGraph(plotImage, pinfo[i].data, xmin, xmax, ymin, ymax, pinfo[i].color, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].thickness);
	}
	makeBB(true);

	Mat temp = render.clone();
	if (isKey != 0)
	{
		Mat roi;
		if (isKey == 1)
		{
			roi = render(Rect(render.cols - keyImage.cols - 150, 80, keyImage.cols, keyImage.rows));
		}
		else if (isKey == 4)
		{
			roi = render(Rect(render.cols - keyImage.cols - 150, render.rows - keyImage.rows - 150, keyImage.cols, keyImage.rows));
		}
		else if (isKey == 2)
		{
			roi = render(Rect(160, 80, keyImage.cols, keyImage.rows));
		}
		else if (isKey == 3)
		{
			roi = render(Rect(160, render.rows - keyImage.rows - 150, keyImage.cols, keyImage.rows));
		}
		keyImage.copyTo(roi);
	}
	addWeighted(render, 0.8, temp, 0.2, 0.0, render);
}

void Plot::save(string name)
{
	FILE* fp = fopen(name.c_str(), "w");

	int dmax = (int)pinfo[0].data.size();
	for (int i = 1; i < data_max; i++)
	{
		dmax = max((int)pinfo[i].data.size(), dmax);
	}

	for (int n = 0; n < dmax; n++)
	{
		for (int i = 0; i < data_max; i++)
		{
			if (n < pinfo[i].data.size())
			{
				double x = pinfo[i].data[n].x;
				double y = pinfo[i].data[n].y;
				fprintf(fp, "%f %f ", x, y);
			}
			else
			{
				double x = pinfo[i].data[pinfo[i].data.size() - 1].x;
				double y = pinfo[i].data[pinfo[i].data.size() - 1].y;
				fprintf(fp, "%f %f ", x, y);
			}
		}
		fprintf(fp, "\n");
	}
	cout << "p ";
	for (int i = 0; i < data_max; i++)
	{
		cout << "'" << name << "'" << " u " << 2 * i + 1 << ":" << 2 * i + 2 << " w lp" << ",";
	}
	cout << endl;
	fclose(fp);
}

Scalar Plot::getPseudoColor(uchar val)
{
	int i = val;
	double d = 255.0 / 63.0;
	Scalar ret;

	{//g
		uchar lr[256];
		for (int i = 0; i < 64; i++)
			lr[i] = cvRound(d*i);
		for (int i = 64; i < 192; i++)
			lr[i] = 255;
		for (int i = 192; i < 256; i++)
			lr[i] = cvRound(255 - d*(i - 192));

		ret.val[1] = lr[val];
	}
	{//r
		uchar lr[256];
		for (int i = 0; i < 128; i++)
			lr[i] = 0;
		for (int i = 128; i < 192; i++)
			lr[i] = cvRound(d*(i - 128));
		for (int i = 192; i < 256; i++)
			lr[i] = 255;

		ret.val[0] = lr[val];
	}
	{//b
		uchar lr[256];
		for (int i = 0; i < 64; i++)
			lr[i] = 255;
		for (int i = 64; i < 128; i++)
			lr[i] = cvRound(255 - d*(i - 64));
		for (int i = 128; i < 256; i++)
			lr[i] = 0;
		ret.val[2] = lr[val];
	}
	return ret;
}

static void guiPreviewMouse(int event, int x, int y, int flags, void* param)
{
	Point* ret = (Point*)param;

	if (flags == cv::EVENT_FLAG_LBUTTON)
	{
		ret->x = x;
		ret->y = y;
	}
}

void Plot::plotMat(InputArray src_, string wname, bool isWait, string gnuplotpath)
{
	Mat src = src_.getMat();
	clear();

	if (src.depth() == CV_32F)
		for (int i = 0; i < src.size().area(); i++) push_back(i, src.at<float>(i));
	else if (src.depth() == CV_64F)
		for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<double>(i));
	else if (src.depth() == CV_8U)
		for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<uchar>(i));
	else if (src.depth() == CV_16U)
		for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<ushort>(i));
	else if (src.depth() == CV_16S)
		for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<short>(i));
	else if (src.depth() == CV_32S)
		for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<int>(i));

	plot(wname, isWait, gnuplotpath);
}

void Plot::plot(string wname, bool isWait, string gnuplotpath)
{
	Point pt = Point(0, 0);
	namedWindow(wname);

	plotData(0, false);
	//int ym=ymax;
	//int yn=ymin;
	//createTrackbar("ymax",wname,&ym,ymax*2);
	//createTrackbar("ymin",wname,&yn,ymax*2);
	setMouseCallback(wname, (MouseCallback)guiPreviewMouse, (void*)&pt);
	int key = 0;
	int isKey = 1;
	int gridlevel = 0;
	makeKey(data_max);

	recomputeXYMAXMIN();
	while (key != 'q')
	{
		//ymax=ym+1;
		//ymin=yn;
		plotData(gridlevel, isKey);
		if (isPosition)
		{
			double xx = 0.0;
			double yy = 0.0;
			point2val(pt, &xx, &yy);
			if (pt.x < 0 || pt.y < 0 || pt.x >= render.cols || pt.y >= render.rows)
			{
				pt = Point(0, 0);
				xx = 0.0;
				yy = 0.0;
			}
			string text = format("(%f,%f)", xx, yy);
			putText(render, text, Point(100, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);
		}

		if (isPosition)drawGrid(render, pt, Scalar(180, 180, 255), 1, 4, 0);

		if (isWait) imshow(wname, render);
		key = waitKey(1);

		if (key == '?')
		{
			cout << "*** Help message ***" << endl;
			cout << "m: " << "show mouseover position and grid" << endl;
			cout << "c: " << "(0,0)point must posit center" << endl;
			cout << "g: " << "Show grid" << endl;

			cout << "k: " << "Show key" << endl;

			cout << "x: " << "Set X origin zero " << endl;
			cout << "y: " << "Set Y origin zero " << endl;
			cout << "z: " << "Set XY origin zero " << endl;
			cout << "r: " << "Reset XY max min" << endl;

			cout << "s: " << "Save image (plot.png)" << endl;
			cout << "q: " << "Quit" << endl;

			cout << "********************" << endl;
			cout << endl;
		}
		if (key == 'm')
		{
			isPosition = (isPosition) ? false : true;
		}

		if (key == 'r')
		{
			recomputeXYMAXMIN(false);
		}
		if (key == 'c')
		{
			recomputeXYMAXMIN(true);
		}
		if (key == 'x')
		{
			setXOriginZERO();
		}
		if (key == 'y')
		{
			setYOriginZERO();
		}
		if (key == 'z')
		{
			setXYOriginZERO();
		}
		if (key == 'k')
		{
			isKey++;
			if (isKey == 5)
				isKey = 0;
		}
		if (key == 'g')
		{
			gridlevel++;
			if (gridlevel > 3)gridlevel = 0;
		}
		if (key == 'p')
		{
			save("plot");
		}
		if (key == 's')
		{
			save("plot");
			imwrite("plotim.png", render);
			//imwrite("plot.png", save);
		}

		if (key == '0')
		{
			for (int i = 0; i < pinfo[0].data.size(); i++)
			{
				cout << i << pinfo[0].data[i] << endl;
			}
		}

		if (!isWait) break;
	}

	if (isWait) destroyWindow(wname);
}


void drawSignalX(InputArray src_, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size outputImageSize, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	vector<Mat> src;
	src_.getMatVector(src);

	Plot p(outputImageSize);
	p.setPlotProfile(false, false, false);
	p.setPlotSymbolALL(Plot::SYMBOL_NOPOINT);
	p.setPlotLineTypeALL(linetype);

	p.setXYMinMax(shiftx - max(rangex, 1), shiftx + max(rangex, 1), shiftvalue - rangevalue, shiftvalue + rangevalue);
	vector<vector<Point>> v((int)src.size());

	for (int i = 0; i < (int)src.size(); i++)
	{
		getImageLine(src[i], v[i], line_height, color);
		p.push_back(v[i], i);
		p.plotData();
	}

	p.render.copyTo(dest);
}

void drawSignalX(Mat& src1, Mat& src2, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	vector<Mat> s;
	s.push_back(src1);
	s.push_back(src2);
	drawSignalX(s, color, dest, size, line_height, shiftx, shiftvalue, rangex, rangevalue, linetype);
}

void drawSignalY(vector<Mat>& src, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	Plot p(size);

	p.setPlotProfile(false, false, false);
	p.setPlotSymbolALL(Plot::SYMBOL_NOPOINT);
	p.setPlotLineTypeALL(linetype);
	p.setXYMinMax(shiftx - max(rangex, 1), shiftx + max(rangex, 1), shiftvalue - rangevalue, shiftvalue + rangevalue);
	vector<vector<Point>> v((int)src.size());
	for (int i = 0; i < (int)src.size(); i++)
	{
		getImageVLine(src[i], v[i], line_height, color);
		p.push_back(v[i], i);
		p.plotData();
	}
	p.render.copyTo(dest);
}

void drawSignalY(Mat& src, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	vector<Mat> s;
	s.push_back(src);
	drawSignalY(s, color, dest, size, line_height, shiftx, shiftvalue, rangex, rangevalue, linetype);
}

void drawSignalY(Mat& src1, Mat& src2, DRAW_SIGNAL_CHANNEL color, Mat& dest, Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype)
{
	vector<Mat> s;
	s.push_back(src1);
	s.push_back(src2);
	drawSignalY(s, color, dest, size, line_height, shiftx, shiftvalue, rangex, rangevalue, linetype);
}

void imshowAnalysis(String winname, Mat& src)
{
	static bool isFirst = true;
	Mat im;
	if (src.channels() == 1)cvtColor(src, im, COLOR_GRAY2BGR);
	else src.copyTo(im);

	namedWindow(winname);
	if (isFirst)moveWindow(winname.c_str(), src.cols * 2, 0);

	static Point pt = Point(src.cols / 2, src.rows / 2);
	static int channel = 0;
	createTrackbar("channel", winname, &channel, 3);
	createTrackbar("x", winname, &pt.x, src.cols - 1);
	createTrackbar("y", winname, &pt.y, src.rows - 1);
	static int step = src.cols / 2;
	createTrackbar("clip x", winname, &step, src.cols / 2);
	static int ystep = src.rows / 2;
	createTrackbar("clip y", winname, &ystep, src.rows / 2);

	string winnameSigx = winname + " Xsignal";
	namedWindow(winnameSigx);

	if (isFirst)moveWindow(winnameSigx.c_str(), 512, src.rows * 2);

	static int shifty = 128;
	createTrackbar("shift y", winnameSigx, &shifty, 128);
	static int stepy = 128;
	createTrackbar("clip y", winnameSigx, &stepy, 255);

	string winnameSigy = winname + " Ysignal";
	namedWindow(winnameSigy);
	if (isFirst)moveWindow(winnameSigy.c_str(), 0, 0);
	static int yshifty = 128;
	createTrackbar("shift y", winnameSigy, &yshifty, 128);
	static int ystepy = 128;
	createTrackbar("clip y", winnameSigy, &ystepy, 255);

	string winnameHist = winname + " Histogram";
	namedWindow(winnameHist);

	Mat dest;
	drawSignalX(src, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.cols, 350), pt.y, pt.x, shifty, step, stepy, 1);
	imshow(winnameSigx, dest);
	drawSignalY(src, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.rows, 350), pt.x, pt.y, yshifty, ystep, ystepy);
	Mat temp;
	flip(dest.t(), temp, 0);
	imshow(winnameSigy, temp);

	rectangle(im, Point(pt.x - step, pt.y - ystep), Point(pt.x + step, pt.y + ystep), COLOR_GREEN);
	drawGrid(im, pt, COLOR_RED);
	imshow(winname, im);

	isFirst = false;
}

void imshowAnalysis(String winname, vector<Mat>& s)
{
	Mat src = s[0];
	static bool isFirst = true;
	vector<Mat> im(s.size());
	for (int i = 0; i < (int)s.size(); i++)
	{
		if (src.channels() == 1)cvtColor(s[i], im[i], COLOR_GRAY2BGR);
		else s[i].copyTo(im[i]);
	}

	namedWindow(winname);
	if (isFirst)moveWindow(winname.c_str(), src.cols * 2, 0);

	static Point pt = Point(src.cols / 2, src.rows / 2);

	static int amp = 1;
	createTrackbar("amp", winname, &amp, 255);
	static int nov = 0;
	createTrackbar("num of view", winname, &nov, (int)s.size() - 1);

	static int channel = 0;
	createTrackbar("channel", winname, &channel, 3);
	createTrackbar("x", winname, &pt.x, src.cols - 1);
	createTrackbar("y", winname, &pt.y, src.rows - 1);
	static int step = src.cols / 2;
	createTrackbar("clip x", winname, &step, src.cols / 2);
	static int ystep = src.rows / 2;
	createTrackbar("clip y", winname, &ystep, src.rows / 2);

	string winnameSigx = winname + " Xsignal";
	namedWindow(winnameSigx);

	if (isFirst)moveWindow(winnameSigx.c_str(), 512, src.rows);

	static int shifty = 128;
	createTrackbar("shift y", winnameSigx, &shifty, 128);
	static int stepy = 128;
	createTrackbar("clip y", winnameSigx, &stepy, 255);

	string winnameSigy = winname + " Ysignal";
	namedWindow(winnameSigy);
	if (isFirst)moveWindow(winnameSigy.c_str(), 0, 0);
	static int yshifty = 128;
	createTrackbar("shift y", winnameSigy, &yshifty, 128);
	static int ystepy = 128;
	createTrackbar("clip y", winnameSigy, &ystepy, 255);

	Mat dest;

	drawSignalX(s, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.cols, 350), pt.y, pt.x, shifty, step, stepy, 1);
	imshow(winnameSigx, dest);
	drawSignalY(s, (DRAW_SIGNAL_CHANNEL)channel, dest, Size(src.rows, 350), pt.x, pt.y, yshifty, ystep, ystepy, 1);
	Mat temp;
	flip(dest.t(), temp, 0);
	imshow(winnameSigy, temp);

	Mat show;
	im[nov].convertTo(show, -1, max(amp, 1));

	//crop
	{
		int x = max(0, pt.x - step);
		int y = max(0, pt.y - ystep);
		int w = min(show.cols, x + 2 * step) - x;
		int h = min(show.rows, y + 2 * ystep) - y;
		Mat rectimage = Mat(show(Rect(x, y, w, h)));
		imshow("crop", rectimage);
	}
	rectangle(show, Point(pt.x - step, pt.y - ystep), Point(pt.x + step, pt.y + ystep), COLOR_GREEN);
	drawGrid(show, pt, COLOR_RED);
	imshow(winname, show);

	isFirst = false;
}