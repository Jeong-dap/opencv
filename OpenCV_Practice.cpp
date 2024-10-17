#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef struct {
	int r, g, b;
}int_rgb;


int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}


float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); //

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B
		+ (1.0 - dx) * dy * C + dx * dy * D;

	return((int)(value + 0.5));
}


void DrawHistogram(char* comments, int* Hist)
{
	int histSize = 256; /// Establish the number of bins
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat r_hist(histSize, 1, CV_32FC1);
	for (int i = 0; i < histSize; i++)
		r_hist.at<float>(i, 0) = Hist[i];
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);

}
int ex0903_1() {
	Mat img = imread("star.png");
	imshow("별", img);
	waitKey(60000);
	return 0;
}
// 십자 모양 만들기
void ex0911_1() {
	int height = 512, width = 1024;
	int** img = (int**)IntAlloc2(height, width);

	int y = 256;
	int x = 512;

	for (x = 0; x < width; x++) {
		img[y][x] = 255;
	}
	x = 512;
	for (y = 0; y < height; y++) {
		img[y][x] = 255;
	}


	ImageShow((char*)"output", img, height, width);

	IntFree2(img, height, width);
}
// 색칠된 직사각형
void ex0911_2() {
	int height = 512, width = 1024;
	int** img = (int**)IntAlloc2(height, width);

	int y, x;

	for (x = 300; x < 700; x++) {
		for (y = 50; y < 450; y++) {
			img[y][x] = 255;
		}
	}

	ImageShow((char*)"output", img, height, width);

	IntFree2(img, height, width);
}
// 함수 만들기
void drawLine(int** img, int y, int x0, int x1) {
	{
		for (int x = x0; x < x1; x++) {
			img[y][x] = 255;
		}
	}
}

void ex0911_3() {
	int height = 512, width = 1024;
	int** img = (int**)IntAlloc2(height, width);
	int y, x;
	int x0 = width * 1 / 4, x1 = width * 3 / 4;
	for (y = 150; y < 406; y++) {
		drawLine(img, y, x0, x1);
	}

	WriteImage((char*)"test.jpg", img, height, width);
	ImageShow((char*)"output", img, height, width);

	IntFree2(img, height, width);

}

void Thresholding(int threshold, int** img, int height, int width, int** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] > threshold) { img_out[y][x] = 255; }
			else { img_out[y][x] = 0; }
		}
	}
}

void ex0924_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int threshold = 128;
	ImageShow((char*)"input", img, height, width);
	for (threshold = 50; threshold < 250; threshold += 50) {
		Thresholding(threshold, img, height, width, img_out);
		ImageShow((char*)"output", img_out, height, width);
	}
}

void ShiftImage(int value, int** img, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img[y][x] = img[y][x] + value;
		}
	}
}

#define GetMax(x, y) ((x > y) ? x : y)
#define GetMin(x, y) ((x < y) ? x : y)

void ClippingImage(int** img, int** img_out, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			/*if (img[y][x] > 255) { img_out[y][x] = 255; }
			else if (img[y][x] < 0) { img_out[y][x] = 0; }
			else { img_out[y][x] = img[y][x]; }*/
			/*int A = GetMax(img[y][x], 0);
			int B = GetMin(A, 255);
			img_out[y][x] = B;*/

			img_out[y][x] = GetMin(GetMax(img[y][x], 0), 255);
		}
	}
}

void ex0924_2() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	ShiftImage(-50, img, height, width);

	ClippingImage(img, img_out, height, width);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

#define NUM 300

void ex0924_3() {
	int aaa = -10, bbb = NUM;
	int A, B;

	A = GetMax(aaa, 0);		// A = ((aaa > 0) ? aaa : 0);
	B = GetMin(A, 255);		// B = ((bbb < 255) ? bbb : 255);
}

void ex0925_1() {
	int A = 100, B = 200, C = 300;
	int max_value = GetMax(GetMax(A, B), C);
}

int FindMaxValue(int** img, int height, int width) {		// 픽셀 최대값 찾기
	int max_value = img[0][0];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			max_value = GetMax(max_value, img[y][x]);
		}
	}
	return max_value;
}

int FindMinValue(int** img, int height, int width) {		// 픽셀 최소값 찾기
	int min_value = img[0][0];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			min_value = GetMin(min_value, img[y][x]);
		}
	}
	return min_value;
}

void ex0925_2() {
	int A[7] = { 1, -1, 3, 8, 2, 9, 10};
	int max_value = A[0];
	int min_value = A[0];
	for (int i = 0; i < 7; i++) {
		max_value = GetMax(max_value, A[i + 1]);
	}

	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	max_value = FindMaxValue(img, height, width);
	min_value = FindMinValue(img, height, width);
}

void MixingImages(float alpha,	// 가중치 ///////////	이미지 섞기		/////////////
	int** img1,					// 첫번째 영상
	int** img2,					// 두번째 영상
	int height,					// 높이
	int width,					// 폭
	int** img_out)	{			// 출력 영상

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = alpha * img1[y][x] + (1.0 - alpha) * img2[y][x];
		}
	}
}


void ex0925_3() {
	int height, width;
	int** img1 = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img2 = ReadImage((char*)"./TestImages/barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	
	float alpha = 0.5;
	for (alpha = 0.0; alpha <= 1.0; alpha += 0.1){
		MixingImages(alpha, img1, img2, height, width, img_out);
		ImageShow((char*)"output", img_out, height, width);
	}
}

void Stretch_1(int** img, int** img_out, int height, int width, int a) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] < a) { img_out[y][x] = (255.0 / a) * img[y][x] + 0.5; }
			else { img_out[y][x] = 255; }
		}
	}
}

void Stretch_2(int a, int b, int c, int d, int** img, int** img_out, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] < a) {
				img_out[y][x] = ((float)c / a) * img[y][x] + 0.5;
			}
			else if (a <= img[y][x] && img[y][x] < b) {
				img_out[y][x] = ((float)d - c) / (b - a) * (img[y][x] - a) + c + 0.5; 
			}
			else {
				img_out[y][x] = (255.0 - d) / (255 - b) * (img[y][x] - b) + d + 0.5;
			}
		}
	}
}

struct Parameter {		// 구조체 정의
	int a, b, c, d;
};

void Stretch_3(Parameter param, int** img, int** img_out, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] < param.a) {
				img_out[y][x] = ((float)param.c / param.a) * img[y][x] + 0.5;
			}
			else if (param.a <= img[y][x] && img[y][x] < param.b) {
				img_out[y][x] = ((float)param.d - param.c) / (param.b - param.a) * (img[y][x] - param.a) + param.c + 0.5;
			}
			else {
				img_out[y][x] = (255.0 - param.d) / (255 - param.b) * (img[y][x] - param.b) + param.d + 0.5;
			}
		}
	}
}

struct ParameterAll {
	int a, b, c, d;
	int** img;
	int** img_out;
	int height, width;
};

void Stretch_4(ParameterAll p) {
	for (int y = 0; y < p.height; y++) {
		for (int x = 0; x < p.width; x++) {
			if (p.img[y][x] < p.a) {
				p.img_out[y][x] = ((float)p.c / p.a) * p.img[y][x] + 0.5;
			}
			else if (p.a <= p.img[y][x] && p.img[y][x] < p.b) {
				p.img_out[y][x] = ((float)p.d - p.c) / (p.b - p.a) * (p.img[y][x] - p.a) + p.c + 0.5;
			}
			else {
				p.img_out[y][x] = (255.0 - p.d) / (255 - p.b) * (p.img[y][x] - p.b) + p.d + 0.5;
			}
		}
	}
}

void ex1002_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	/*int a = 150;
	Stretch_1(img, img_out, height, width, a);*/

	/*int a = 100, b = 150, c = 50, d = 200;
	Stretch_2(a, b, c, d, img, img_out, height, width);*/

	//Parameter param;	// 구조체 사용
	//param.a = 100; param.b = 150; param.c = 50; param.d = 200;
	//Stretch_3(param, img, img_out, height, width);

	ParameterAll param;
	param.a = 100; param.b = 150; param.c = 50; param.d = 200;
	param.img = img; param.img_out = img_out; param.height = height; param.width = width;
	Stretch_4(param);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

void round_up() {		// 반올림
	float a = 100.5;		// 100.5
	int b = a;				// 100
}

int GetCount(int** img, int value, int height, int width) {
	
	int count = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] == value) {
				count++;
			}
		}
	}
	return count;
}

void GetHistogram(int** img, int* histogram, int height, int width) {
	for (int value = 0; value < 256; value++) {
		histogram[value] = GetCount(img, value, height, width);
	}
}

void ex1008_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);

	int histogram[256];
	GetHistogram(img, histogram, height, width);
	DrawHistogram((char*)"histo", histogram);
}

void GetHistogram2(int** img, int* histogram, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			histogram[img[y][x]]++;
		}
	}
}

void GetChistogram(int** img, int* chist, int height, int width) {
	int histogram[256] = { 0 };
	GetHistogram2(img, histogram, height, width);

	chist[0] = histogram[0];
	for (int n = 1; n < 256; n++) {
		chist[n] = chist[n - 1] + histogram[n];
	}
}

void HistogramEQ(int** img, int** img_out, int height, int width) {
	int chist[256] = { 0 };
	GetChistogram(img, chist, height, width);

	int norm_chist[256] = { 0 };
	for (int n = 0; n < 256; n++) {
		norm_chist[n] = (float)chist[n] / (width * height) * 255 + 0.5;
	}

	// mapping using 'norm chist[]'
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = norm_chist[img[y][x]];
		}
	}
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	int hist_input[256] = { 0 }; int hist_output[256] = { 0 };
	GetHistogram2(img, hist_input, height, width);
	GetHistogram2(img_out, hist_output, height, width);
	DrawHistogram((char*)"input_hist", hist_input);
	DrawHistogram((char*)"output_hist", hist_output);
}

void ex1008_2() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lenax0.5.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	HistogramEQ(img, img_out, height, width);
}


///////////////////////////////////////////////////////////////////////////////////////
void sample() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
}

int getMean3x3(int y, int x, int** img) {
	int sum = 0;
	for (int m = -1; m <= 1; m++) {
		for (int n = -1; n <= 1; n++) {
			sum += img[y + m][x + n];
		}
	}
	return (int)(sum / 9.0 + 0.5);
}

void MeanFilter3x3(int** img, int** img_out, int height, int width) {
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			img_out[y][x] = getMean3x3(y, x, img);
		}
	}

	for (int x = 0; x < width; x++) {
		int y = 0;
		img_out[y][x] = img[y][x];
	}
	for (int x = 0; x < width; x++) {
		int y = height - 1;
		img_out[y][x] = img[y][x];
	}
	for (int y = 0; y < height; y++) {
		int x = 0;
		img_out[y][x] = img[y][x];
	}
	for (int y = 0; y < height; y++) {
		int x = width - 1;
		img_out[y][x] = img[y][x];
	}
}

int getMean5x5(int y, int x, int** img) {
	int sum = 0;
	for (int m = -2; m <= 2; m++) {
		for (int n = -2; n <= 2; n++) {
			sum += img[y + m][x + n];
		}
	}
	return (int)(sum / 25.0 + 0.5);
}

void MeanFilter5x5(int** img, int** img_out, int height, int width) {
	for (int y = 2; y < height - 2; y++) {
		for (int x = 2; x < width - 2; x++) {
			img_out[y][x] = getMean5x5(y, x, img);
		}
	}
	for (int y = 0; y < 2; y++) {
		for (int x = 0; x < width; x++) { img_out[y][x] = img[y][x]; }
	}
	for (int y = height - 2; y < height; y++) {
		for (int x = 0; x < width; x++) { img_out[y][x] = img[y][x]; }
	}
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < height; y++) { img_out[y][x] = img[y][x]; }
	}
	for (int x = width - 2; x < width; x++){
		for (int y = 0; y < height; y++) { img_out[y][x] = img[y][x]; }
	}
}

int getMean7x7(int y, int x, int** img) {
	int sum = 0;
	for (int m = -3; m <= 3; m++) {
		for (int n = -3; n <= 3; n++) {
			sum += img[y + m][x + n];
		}
	}
	return (int)(sum / 49.0 + 0.5);
}

void MeanFilter7x7(int** img, int** img_out, int height, int width) {
	for (int y = 3; y < height - 3; y++) {
		for (int x = 3; x < width - 3; x++) {
			img_out[y][x] = getMean7x7(y, x, img);
		}
	}
	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < width; x++) { img_out[y][x] = img[y][x]; }
	}
	for (int y = height - 3; y < height; y++) {
		for (int x = 0; x < width; x++) { img_out[y][x] = img[y][x]; }
	}
	for (int x = 0; x < 3; x++) {
		for (int y = 0; y < height; y++) { img_out[y][x] = img[y][x]; }
	}
	for (int x = width - 3; x < width; x++) {
		for (int y = 0; y < height; y++) { img_out[y][x] = img[y][x]; }
	}
}

void ex1015_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	MeanFilter3x3(img, img_out, height, width);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

int getMeanNxN(int N, int y, int x, int** img) {
	int sum = 0;
	int value = (N - 1) / 2;
	for (int m = -value; m <= value; m++) {
		for (int n = -value; n <= value; n++) {
			sum += img[y + m][x + n];
		}
	}
	return (int)(sum / (N * N) + 0.5);
}

void MeanFilterNxN(int N, int** img, int** img_out, int height, int width) {
	int value = (N - 1) / 2;
	for (int y = value; y < height - value; y++) {
		for (int x = value; x < width - value; x++) {
			img_out[y][x] = getMeanNxN(N, y, x, img);
		}
	}
	for (int y = 0; y < value; y++) {
		for (int x = 0; x < width; x++) { img_out[y][x] = img[y][x]; }
	}
	for (int y = height - value; y < height; y++) {
		for (int x = 0; x < width; x++) { img_out[y][x] = img[y][x]; }
	}
	for (int x = 0; x < value; x++) {
		for (int y = 0; y < height; y++) { img_out[y][x] = img[y][x]; }
	}
	for (int x = width - value; x < width; x++) {
		for (int y = 0; y < height; y++) { img_out[y][x] = img[y][x]; }
	}
}


void ex1016_1() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out3x3 = (int**)IntAlloc2(height, width);
	int** img_out5x5 = (int**)IntAlloc2(height, width);
	int** img_out7x7 = (int**)IntAlloc2(height, width);
	int** img_outnxn = (int**)IntAlloc2(height, width);

	for (int N = 50; N < 60; N++) {
		MeanFilterNxN(N, img, img_outnxn, height, width);
		ImageShow((char*)"outputnxn", img_outnxn, height, width);
	}
}

void main() {
	int height, width;
	int** img = ReadImage((char*)"./TestImages/lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	float** kernel = (float**)FloatAlloc2(height, width);

	kernel[0][0] = 1 / 9.0;  kernel[0][1] = 1 / 9.0;   kernel[0][2] = 1 / 9.0;
	kernel[1][0] = 1 / 9.0;  kernel[1][1] = 1 / 9.0;   kernel[1][2] = 1 / 9.0;
	kernel[2][0] = 1 / 9.0;  kernel[2][1] = 1 / 9.0;   kernel[2][2] = 1 / 9.0;

	int y = 100, x = 200;

	float sum = 0.0;

	for (int m = -1; m <= 1; m++) {
		for (int n = -1; n <= 1; n++) {
			sum += img[y + m][x + n] * kernel[m + 1][n + 1];
		}
	}
	img_out[y][x] = sum + 0.5;
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}