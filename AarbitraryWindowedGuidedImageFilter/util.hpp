#pragma once

#include <opencv2/core.hpp>
#define COLOR_WHITE cv::Scalar(255,255,255)
#define COLOR_GRAY10 cv::Scalar(10,10,10)
#define COLOR_GRAY20 cv::Scalar(20,20,20)
#define COLOR_GRAY30 cv::Scalar(10,30,30)
#define COLOR_GRAY40 cv::Scalar(40,40,40)
#define COLOR_GRAY50 cv::Scalar(50,50,50)
#define COLOR_GRAY60 cv::Scalar(60,60,60)
#define COLOR_GRAY70 cv::Scalar(70,70,70)
#define COLOR_GRAY80 cv::Scalar(80,80,80)
#define COLOR_GRAY90 cv::Scalar(90,90,90)
#define COLOR_GRAY100 cv::Scalar(100,100,100)
#define COLOR_GRAY110 cv::Scalar(101,110,110)
#define COLOR_GRAY120 cv::Scalar(120,120,120)
#define COLOR_GRAY130 cv::Scalar(130,130,140)
#define COLOR_GRAY140 cv::Scalar(140,140,140)
#define COLOR_GRAY150 cv::Scalar(150,150,150)
#define COLOR_GRAY160 cv::Scalar(160,160,160)
#define COLOR_GRAY170 cv::Scalar(170,170,170)
#define COLOR_GRAY180 cv::Scalar(180,180,180)
#define COLOR_GRAY190 cv::Scalar(190,190,190)
#define COLOR_GRAY200 cv::Scalar(200,200,200)
#define COLOR_GRAY210 cv::Scalar(210,210,210)
#define COLOR_GRAY220 cv::Scalar(220,220,220)
#define COLOR_GRAY230 cv::Scalar(230,230,230)
#define COLOR_GRAY240 cv::Scalar(240,240,240)
#define COLOR_GRAY250 cv::Scalar(250,250,250)
#define COLOR_RED cv::Scalar(0,0,255)
#define COLOR_GREEN cv::Scalar(0,255,0)
#define COLOR_BLUE cv::Scalar(255,0,0)
#define COLOR_ORANGE cv::Scalar(0,100,255)
#define COLOR_YELLOW cv::Scalar(0,255,255)
#define COLOR_MAGENDA cv::Scalar(255,0,255)
#define COLOR_CYAN cv::Scalar(255,255,0)
#define COLOR_BLACK cv::Scalar(0,0,0)

enum
{
	TIME_AUTO = 0,
	TIME_NSEC,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR,
	TIME_DAY
};

class CalcTime
{
	int64 pre;
	std::string mes;

	int timeMode;

	double cTime;
	bool _isShow;

	int autoMode;
	int autoTimeMode();
	std::vector<std::string> lap_mes;
public:

	void start();
	void setMode(int mode);
	void setMessage(std::string& src);
	void restart();
	double getTime();
	void show();
	void show(std::string message);
	void lap(std::string message);
	void init(std::string message, int mode, bool isShow);

	CalcTime(std::string message, int mode = TIME_AUTO, bool isShow = true);
	CalcTime(char* message, int mode = TIME_AUTO, bool isShow = true);
	CalcTime();

	~CalcTime();
};

class ConsoleImage
{
private:
	int count;
	std::string windowName;
	std::vector<std::string> strings;
	bool isLineNumber;
public:
	void setIsLineNumber(bool isLine = true);
	bool getIsLineNumber();
	cv::Mat show;

	void init(cv::Size size, std::string wname);
	ConsoleImage();
	ConsoleImage(cv::Size size, std::string wname = "console");
	~ConsoleImage();

	void printData();
	void clear();

	void operator()(std::string src);
	void operator()(const char *format, ...);
	void operator()(cv::Scalar color, const char *format, ...);

	void flush(bool isClear = true);
};

class Stat
{
public:
	std::vector<double> data;
	int num_data;
	Stat();
	~Stat();
	double getMin();
	double getMax();
	double getMean();
	double getStd();
	double getMedian();

	void push_back(double val);

	void clear();
	void show();
};

class Plot
{
protected:
	struct PlotInfo
	{
		std::vector<cv::Point2d> data;
		cv::Scalar color;
		int symbolType;
		int lineType;
		int thickness;

		std::string keyname;
	};
	std::vector<PlotInfo> pinfo;

	std::string xlabel;
	std::string ylabel;

	int data_max;

	cv::Scalar background_color;

	cv::Size plotsize;
	cv::Point origin;

	double xmin;
	double xmax;
	double ymin;
	double ymax;
	double xmax_no_margin;
	double xmin_no_margin;
	double ymax_no_margin;
	double ymin_no_margin;

	void init();
	void point2val(cv::Point pt, double* valx, double* valy);

	bool isZeroCross;
	bool isXYMAXMIN;
	bool isXYCenter;

	bool isPosition;
	cv::Scalar getPseudoColor(uchar val);
	cv::Mat plotImage;
	cv::Mat keyImage;
public:
	//symbolType
	enum
	{
		SYMBOL_NOPOINT = 0,
		SYMBOL_PLUS,
		SYMBOL_TIMES,
		SYMBOL_ASTERRISK,
		SYMBOL_CIRCLE,
		SYMBOL_RECTANGLE,
		SYMBOL_CIRCLE_FILL,
		SYMBOL_RECTANGLE_FILL,
		SYMBOL_TRIANGLE,
		SYMBOL_TRIANGLE_FILL,
		SYMBOL_TRIANGLE_INV,
		SYMBOL_TRIANGLE_INV_FILL,
	};

	//lineType
	enum
	{
		LINE_NONE,
		LINE_LINEAR,
		LINE_H2V,
		LINE_V2H
	};

	cv::Mat render;
	cv::Mat graphImage;

	Plot(cv::Size window_size = cv::Size(1024, 768));
	~Plot();

	void setXYOriginZERO();
	void setXOriginZERO();
	void setYOriginZERO();

	void recomputeXYMAXMIN(bool isCenter = false, double marginrate = 0.9);
	void setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_, bool isZeroCross_);
	void setPlotImageSize(cv::Size s);
	void setXYMinMax(double xmin_, double xmax_, double ymin_, double ymax_);
	void setXMinMax(double xmin_, double xmax_);
	void setYMinMax(double ymin_, double ymax_);
	void setBackGoundColor(cv::Scalar cl);

	void makeBB(bool isFont);

	void setPlot(int plotnum, cv::Scalar color = COLOR_RED, int symboltype = SYMBOL_PLUS, int linetype = LINE_LINEAR, int thickness = 1);
	void setPlotThickness(int plotnum, int thickness_);
	void setPlotColor(int plotnum, cv::Scalar color);
	void setPlotSymbol(int plotnum, int symboltype);
	void setPlotLineType(int plotnum, int linetype);
	void setPlotKeyName(int plotnum, std::string name);

	void setPlotSymbolALL(int symboltype);
	void setPlotLineTypeALL(int linetype);

	void plotPoint(cv::Point2d = cv::Point2d(0.0, 0.0), cv::Scalar color = COLOR_BLACK, int thickness_ = 1, int linetype = LINE_LINEAR);
	void plotGrid(int level);
	void plotData(int gridlevel = 0, int isKey = 0);

	void plotMat(cv::InputArray src, std::string name = "Plot", bool isWait = true, std::string gnuplotpath = "pgnuplot.exe");
	void plot(std::string name = "Plot", bool isWait = true, std::string gnuplotpath = "pgnuplot.exe");

	void makeKey(int num);

	void save(std::string name);

	void push_back(std::vector<cv::Point> point, int plotIndex = 0);
	void push_back(std::vector<cv::Point2d> point, int plotIndex = 0);
	void push_back(double x, double y, int plotIndex = 0);

	void erase(int sampleIndex, int plotIndex = 0);
	void insert(cv::Point2d v, int sampleIndex, int plotIndex = 0);
	void insert(cv::Point v, int sampleIndex, int plotIndex = 0);
	void insert(double x, double y, int sampleIndex, int plotIndex = 0);

	void clear(int datanum = -1);

	void swapPlot(int plotIndex1, int plotIndex2);
};

void alphaBlend(cv::InputArray src1, cv::InputArray src2, const double alpha, cv::OutputArray dest);
double calcPSNR(cv::InputArray I1, cv::InputArray I2);
double calcPSNR(cv::Mat& src1, cv::Mat& src2, cv::Rect rect);
void addNoise(cv::InputArray src, cv::OutputArray dest, double sigma, double sprate = 0.0);
void triangle(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar& color, int thickness = 1);
void triangleinv(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar& color, int thickness = 1);
void drawPlus(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
void drawTimes(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_typee = 8, int shift = 0);
void drawGrid(cv::InputOutputArray src, cv::Point crossCenter, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
void drawAsterisk(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);

void plotGraph(cv::OutputArray graph, std::vector<cv::Point2d>& data, double xmin, double xmax, double ymin, double ymax,
	cv::Scalar color = COLOR_RED, int lt = Plot::SYMBOL_PLUS, int isLine = Plot::LINE_LINEAR, int thickness = 1, int ps = 4);

void imshowAnalysis(cv::String winname, std::vector<cv::Mat>& s);
void imshowAnalysis(cv::String winname, cv::Mat& src);

enum DRAW_SIGNAL_CHANNEL
{
	B,
	G,
	R,
	Y
};
void drawSignalX(cv::Mat& src1, cv::Mat& src2, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size outputImageSize, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = Plot::LINE_LINEAR);// color 0:B, 1:G, 2:R, 3:Y
void drawSignalX(cv::InputArray src, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size outputImageSize, int analysisLineHeight, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = Plot::LINE_LINEAR);// color 0:B, 1:G, 2:R, 3:Y

void drawSignalY(cv::Mat& src1, cv::Mat& src2, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = Plot::LINE_LINEAR);// color 0:B, 1:G, 2:R, 3:Y
void drawSignalY(cv::Mat& src, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = Plot::LINE_LINEAR);// color 0:B, 1:G, 2:R, 3:Y
void drawSignalY(std::vector<cv::Mat>& src, DRAW_SIGNAL_CHANNEL color, cv::Mat& dest, cv::Size size, int line_height, int shiftx, int shiftvalue, int rangex, int rangevalue, int linetype = Plot::LINE_LINEAR);// color 0:B, 1:G, 2:R, 3:Y
