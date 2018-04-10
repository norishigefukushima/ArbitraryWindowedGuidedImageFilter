#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

enum
{
	VECTOR_WITHOUT = 0,
	VECTOR_AVX = 1
};

float sigma2LaplacianSmootihngAlpha(const float sigma, float p)
{
	return 1.f - exp(-p / (1.f*sigma));
}

void LaplacianSmoothingFIR2DFilter(Mat& src, Mat& dest, float sigma)
{
	int r = cvRound(5.f*sigma);
	Mat kernel = Mat::zeros(2 * r + 1, 2 * r + 1, CV_32F);
	float total = 0.f;
	for (int j = 0; j < kernel.rows; j++)
	{
		for (int i = 0; i < kernel.cols; i++)
		{
			float p = (float)abs(sqrt((i - r)*(i - r) + (j - r)*(j - r)));
			float v = exp(-p / sigma);
			kernel.at<float>(j, i) = v;
			total += v;
		}
	}
	for (int j = 0; j < kernel.rows; j++)
	{
		for (int i = 0; i < kernel.cols; i++)
		{
			kernel.at<float>(j, i) /= total;
		}
	}
	filter2D(src, dest, CV_32F, kernel);
}

void LaplacianSmoothingFIRFilter(Mat& src, Mat& dest, const int r, const float sigma, int border, int opt)
{
	if (dest.empty())dest.create(src.size(), src.type());
	const int ksize = (2 * r + 1);
	Mat im;
	copyMakeBorder(src, im, r, r, r, r, border);

	float* gauss = (float*)_mm_malloc(sizeof(float)*ksize, 32);
	const float gfrac = -1.f / (sigma);
	float gsum = 0.f;
	for (int j = -r, index = 0; j <= r; j++)
	{
		float v = exp(abs(j)*gfrac);
		gsum += v;
		gauss[index] = v;
		index++;
	}
	for (int j = -r, index = 0; j <= r; j++)
	{
		//gauss[index] = max(FLT_EPSILON, gauss[index]/gsum);
		gauss[index] /= gsum;
		index++;
	}

	const int wstep = im.cols;
	if (opt == VECTOR_WITHOUT)
	{
#pragma omp parallel for
		for (int j = 0; j < im.rows; j++)
		{
			float* s = im.ptr<float>(j);
			float* d = im.ptr<float>(j);
			for (int i = 0; i < src.cols; i++)
			{
				float v = 0.f;
				for (int k = 0; k < ksize; k++)
				{
					v += gauss[k] * s[i + k];
				}
				d[i] = v;
			}
		}
#pragma omp parallel for
		for (int j = 0; j < src.rows; j++)
		{
			float* s = im.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 0; i < src.cols; i++)
			{
				float v = 0.f;
				for (int k = 0; k < ksize; k++)
				{
					v += gauss[k] * s[i + k*wstep];
				}
				d[i] = v;
			}
		}
	}
	else if (opt == VECTOR_AVX)
	{
#pragma omp parallel for
		for (int j = 0; j < im.rows; j++)
		{
			float* s = im.ptr<float>(j);
			float* d = im.ptr<float>(j);
			for (int i = 0; i < src.cols; i += 8)
			{
				__m256 mv = _mm256_setzero_ps();
				for (int k = 0; k < ksize; k++)
				{
					__m256 ms = _mm256_loadu_ps(s + i + k);
					__m256 mg = _mm256_set1_ps(gauss[k]);
					mv = _mm256_add_ps(mv, _mm256_mul_ps(ms, mg));
				}
				_mm256_storeu_ps(d + i, mv);
			}
		}

#pragma omp parallel for
		for (int j = 0; j < src.rows; j++)
		{
			float* s = im.ptr<float>(j);
			float* d = dest.ptr<float>(j);
			for (int i = 0; i < src.cols; i += 8)
			{
				__m256 mv = _mm256_setzero_ps();
				for (int k = 0; k < ksize; k++)
				{
					__m256 ms = _mm256_loadu_ps(s + i + k*wstep);
					__m256 mg = _mm256_set1_ps(gauss[k]);
					mv = _mm256_add_ps(mv, _mm256_mul_ps(ms, mg));
				}
				_mm256_storeu_ps(d + i, mv);
			}
		}
	}
	_mm_free(gauss);
}

void LaplacianSmoothingIIRFilterBase(Mat& src, Mat& dest, const double sigma_)
{
	if (dest.empty())dest.create(src.size(), src.type());
	Mat tmp(src.size(), src.type());
	Mat tmp2(src.rows, 1, src.type());
	if (src.depth() == CV_32F)
	{
		const float sigma = (float)sigma_;
		const float is = 1.f - sigma;
		for (int j = 0; j < src.rows; j++)
		{
			float* im = src.ptr<float>(j);
			float* dt = dest.ptr<float>(j);
			float* tp = tmp.ptr<float>(j);
			dt[0] = im[0];
			for (int i = 1; i < src.cols; i++)
			{
				dt[i] = sigma*im[i] + is*dt[i - 1];
			}

			tp[src.cols - 1] = im[src.cols - 1];
			dt[src.cols - 1] = (im[src.cols - 1] + dt[src.cols - 1])*0.5f;
			for (int i = src.cols - 2; i >= 0; i--)
			{
				tp[i] = sigma*im[i] + is*tp[i + 1];
				dt[i] = (dt[i] + tp[i])*0.5f;
			}
		}
		for (int i = 0; i < src.cols; i++)
		{
			float* im = dest.ptr<float>(0) + i;
			float* dt = dest.ptr<float>(0) + i;
			float* tp = tmp.ptr<float>(0) + i;
			float* tp2 = tmp2.ptr<float>(0);
			tp[0] = im[0];
			for (int j = 1; j < src.rows; j++)
			{
				tp[j*src.cols] = sigma*im[j*src.cols] + is*tp[(j - 1)*src.cols];
			}
			tp2[src.rows - 1] = im[src.cols*(src.rows - 1)];
			dt[src.cols*(src.rows - 1)] = (im[src.cols*(src.rows - 1)] + tp[src.cols*(src.rows - 1)])*0.5f;
			for (int j = src.rows - 2; j >= 0; j--)
			{
				tp2[j] = sigma*im[j*src.cols] + is*tp2[j + 1];
				dt[j*src.cols] = (tp[j*src.cols] + tp2[j])*0.5f;
			}
		}
	}
	else if (src.depth() == CV_64F)
	{
		const double sigma = sigma_;
		const double is = 1.f - sigma;

		for (int j = 0; j < src.rows; j++)
		{
			double* im = src.ptr<double>(j);
			double* dt = dest.ptr<double>(j);
			double* tp = tmp.ptr<double>(j);
			dt[0] = im[0];
			for (int i = 1; i < src.cols; i++)
			{
				dt[i] = sigma*im[i] + is*dt[i - 1];
			}

			tp[src.cols - 1] = im[src.cols - 1];
			dt[src.cols - 1] = (im[src.cols - 1] + dt[src.cols - 1])*0.5f;
			for (int i = src.cols - 2; i >= 0; i--)
			{
				tp[i] = sigma*im[i] + is*tp[i + 1];
				dt[i] = (dt[i] + tp[i])*0.5f;
			}
		}
		for (int i = 0; i < src.cols; i++)
		{
			double* im = dest.ptr<double>(0) + i;
			double* dt = dest.ptr<double>(0) + i;
			double* tp = tmp.ptr<double>(0) + i;
			double* tp2 = tmp2.ptr<double>(0);
			tp[0] = im[0];
			for (int j = 1; j < src.rows; j++)
			{
				tp[j*src.cols] = sigma*im[j*src.cols] + is*tp[(j - 1)*src.cols];
			}
			tp2[src.rows - 1] = im[src.cols*(src.rows - 1)];
			dt[src.cols*(src.rows - 1)] = (im[src.cols*(src.rows - 1)] + tp[src.cols*(src.rows - 1)])*0.5;
			for (int j = src.rows - 2; j >= 0; j--)
			{
				tp2[j] = sigma*im[j*src.cols] + is*tp2[j + 1];
				dt[j*src.cols] = (tp[j*src.cols] + tp2[j])*0.5f;
			}
		}
	}
}

void LaplacianSmoothingIIRFilterAVXUnroall8(Mat& src, Mat& dest, const float sigma)
{
	if (dest.empty()) dest.create(src.size(), src.type());
	
	Mat buff(max(src.cols * 8, src.rows * 8), 1, src.type());

	const __m256i gidx = _mm256_set_epi32(0, src.cols, 2 * src.cols, 3 * src.cols, 4 * src.cols, 5 * src.cols, 6 * src.cols, 7 * src.cols);

	for (int j = 0; j < src.rows; j += 8)
	{
		const __m256 ms = _mm256_set1_ps(sigma);
		const __m256 mis = _mm256_set1_ps(1.f - sigma);
		const __m256 half = _mm256_set1_ps(0.5f);
		float* im = src.ptr<float>(j);
		float* dt = dest.ptr<float>(j);
		float* b = buff.ptr<float>(0);

		const int idx0 = 0;
		const int idx1 = 1 * src.cols;
		const int idx2 = 2 * src.cols;
		const int idx3 = 3 * src.cols;
		const int idx4 = 4 * src.cols;
		const int idx5 = 5 * src.cols;
		const int idx6 = 6 * src.cols;
		const int idx7 = 7 * src.cols;

		__m256 pv = _mm256_i32gather_ps(im, gidx, 4);
		_mm256_store_ps(b, pv);
		dt[idx0] = im[idx0];
		dt[idx1] = im[idx1];
		dt[idx2] = im[idx2];
		dt[idx3] = im[idx3];
		dt[idx4] = im[idx4];
		dt[idx5] = im[idx5];
		dt[idx6] = im[idx6];
		dt[idx7] = im[idx7];

		for (int i = 1; i < src.cols; i++)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i, gidx, 4), _mm256_mul_ps(mis, pv));
			_mm256_store_ps(b + 8 * i, pv);
		}
		__m256 t = pv;
		pv = _mm256_i32gather_ps(im + src.cols - 1, gidx, 4);
		t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));
		dt[idx0 + src.cols - 1] = t.m256_f32[7];
		dt[idx1 + src.cols - 1] = t.m256_f32[6];
		dt[idx2 + src.cols - 1] = t.m256_f32[5];
		dt[idx3 + src.cols - 1] = t.m256_f32[4];
		dt[idx4 + src.cols - 1] = t.m256_f32[3];
		dt[idx5 + src.cols - 1] = t.m256_f32[2];
		dt[idx6 + src.cols - 1] = t.m256_f32[1];
		dt[idx7 + src.cols - 1] = t.m256_f32[0];

		for (int i = src.cols - 2; i >= 0; i--)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i, gidx, 4), _mm256_mul_ps(mis, pv));
			//__m256 tmpp = _mm256_i32gather_ps(im + i - 7, gidx, 4);
			__m256 v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * i), pv));
			dt[i + idx0] = v.m256_f32[7];
			dt[i + idx1] = v.m256_f32[6];
			dt[i + idx2] = v.m256_f32[5];
			dt[i + idx3] = v.m256_f32[4];
			dt[i + idx4] = v.m256_f32[3];
			dt[i + idx5] = v.m256_f32[2];
			dt[i + idx6] = v.m256_f32[1];
			dt[i + idx7] = v.m256_f32[0];
		}
	}

	for (int i = 0; i < src.cols; i += 8)
	{
		const __m256 ms = _mm256_set1_ps(sigma);
		const __m256 mis = _mm256_set1_ps(1.f - sigma);
		const __m256 half = _mm256_set1_ps(0.5f);
		float* im = dest.ptr<float>(0) + i;
		float* dt = dest.ptr<float>(0) + i;
		float* b = buff.ptr<float>(0);

		__m256 pv = _mm256_loadu_ps(im);
		_mm256_store_ps(b, pv);
		for (int j = 1; j < src.rows; j++)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_loadu_ps(im + j*src.cols), _mm256_mul_ps(mis, pv));
			_mm256_store_ps(b + j * 8, pv);
		}

		__m256 t = pv;
		pv = _mm256_loadu_ps(im + src.cols*(src.rows - 1));
		t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));
		_mm256_storeu_ps(dt + (src.rows - 1)*src.cols, t);

		for (int j = src.rows - 2; j >= 0; j--)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_loadu_ps(im + j*src.cols), _mm256_mul_ps(mis, pv));
			_mm256_storeu_ps(dt + j*src.cols, _mm256_mul_ps(half, _mm256_add_ps(_mm256_loadu_ps(b + j * 8), pv)));
		}
	}
}

void LaplacianSmoothingIIRFilterAVXUnroall16(Mat& src, Mat& dest, const float sigma)
{
	if (dest.empty())dest.create(src.size(), src.type());
	Mat buff(max(src.cols * 16, src.rows * 16), 1, src.type());

	const __m256i gidx = _mm256_set_epi32(0, src.cols, 2 * src.cols, 3 * src.cols, 4 * src.cols, 5 * src.cols, 6 * src.cols, 7 * src.cols);
	const int uidx = 8 * src.cols;

	for (int j = 0; j < src.rows; j += 16)
	{
		const __m256 ms = _mm256_set1_ps(sigma);
		const __m256 mis = _mm256_set1_ps(1.f - sigma);
		const __m256 half = _mm256_set1_ps(0.5f);
		float* im = src.ptr<float>(j);
		float* im2 = src.ptr<float>(j+8);
		float* dt = dest.ptr<float>(j);
		float* dt2 = dest.ptr<float>(j+8);
		float* b = buff.ptr<float>(0);

		const int idx0 = 0;
		const int idx1 = 1 * src.cols;
		const int idx2 = 2 * src.cols;
		const int idx3 = 3 * src.cols;
		const int idx4 = 4 * src.cols;
		const int idx5 = 5 * src.cols;
		const int idx6 = 6 * src.cols;
		const int idx7 = 7 * src.cols;

		__m256 pv = _mm256_i32gather_ps(im, gidx, 4);
		__m256 pv2 = _mm256_i32gather_ps(im + uidx, gidx, 4);
		_mm256_store_ps(b, pv);
		_mm256_store_ps(b+8, pv2);
		dt[idx0] = im[idx0];
		dt[idx1] = im[idx1];
		dt[idx2] = im[idx2];
		dt[idx3] = im[idx3];
		dt[idx4] = im[idx4];
		dt[idx5] = im[idx5];
		dt[idx6] = im[idx6];
		dt[idx7] = im[idx7];
		dt2[idx0] = im2[idx0];
		dt2[idx1] = im2[idx1];
		dt2[idx2] = im2[idx2];
		dt2[idx3] = im2[idx3];
		dt2[idx4] = im2[idx4];
		dt2[idx5] = im2[idx5];
		dt2[idx6] = im2[idx6];
		dt2[idx7] = im2[idx7];
		for (int i = 1; i < src.cols; i++)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i, gidx, 4), _mm256_mul_ps(mis, pv));
			_mm256_store_ps(b + 16 * i, pv);

			pv2 = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i+uidx, gidx, 4), _mm256_mul_ps(mis, pv2));
			_mm256_store_ps(b + 16 * i+8, pv2);
		}
		__m256 t = pv;
		pv = _mm256_i32gather_ps(im + src.cols - 1, gidx, 4);
		pv2 = _mm256_i32gather_ps(im + src.cols - 1+uidx, gidx, 4);
		t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));
		dt[idx0 + src.cols - 1] = t.m256_f32[7];
		dt[idx1 + src.cols - 1] = t.m256_f32[6];
		dt[idx2 + src.cols - 1] = t.m256_f32[5];
		dt[idx3 + src.cols - 1] = t.m256_f32[4];
		dt[idx4 + src.cols - 1] = t.m256_f32[3];
		dt[idx5 + src.cols - 1] = t.m256_f32[2];
		dt[idx6 + src.cols - 1] = t.m256_f32[1];
		dt[idx7 + src.cols - 1] = t.m256_f32[0];
		for (int i = src.cols - 2; i >= 0; i--)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i, gidx, 4), _mm256_mul_ps(mis, pv));
			__m256 v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 16 * i), pv));
			dt[i + idx0] = v.m256_f32[7];
			dt[i + idx1] = v.m256_f32[6];
			dt[i + idx2] = v.m256_f32[5];
			dt[i + idx3] = v.m256_f32[4];
			dt[i + idx4] = v.m256_f32[3];
			dt[i + idx5] = v.m256_f32[2];
			dt[i + idx6] = v.m256_f32[1];
			dt[i + idx7] = v.m256_f32[0];

			pv2 = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i+uidx, gidx, 4), _mm256_mul_ps(mis, pv2));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 16 * i+8), pv2));
			dt2[i + idx0] = v.m256_f32[7];
			dt2[i + idx1] = v.m256_f32[6];
			dt2[i + idx2] = v.m256_f32[5];
			dt2[i + idx3] = v.m256_f32[4];
			dt2[i + idx4] = v.m256_f32[3];
			dt2[i + idx5] = v.m256_f32[2];
			dt2[i + idx6] = v.m256_f32[1];
			dt2[i + idx7] = v.m256_f32[0];
		}
	}

	for (int i = 0; i < src.cols; i += 8)
	{
		const __m256 ms = _mm256_set1_ps(sigma);
		const __m256 mis = _mm256_set1_ps(1.f - sigma);
		const __m256 half = _mm256_set1_ps(0.5f);
		float* im = dest.ptr<float>(0) + i;
		float* dt = dest.ptr<float>(0) + i;
		float* b = buff.ptr<float>(0);

		__m256 pv = _mm256_load_ps(im);
		_mm256_storeu_ps(b, pv);
		for (int j = 1; j < src.rows; j++)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_loadu_ps(im + j*src.cols), _mm256_mul_ps(mis, pv));
			_mm256_store_ps(b + j * 8, pv);
		}

		__m256 t = pv;
		pv = _mm256_loadu_ps(im + src.cols*(src.rows - 1));
		t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));
		_mm256_storeu_ps(dt + (src.rows - 1)*src.cols, t);

		for (int j = src.rows - 2; j >= 0; j--)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_loadu_ps(im + j*src.cols), _mm256_mul_ps(mis, pv));
			_mm256_storeu_ps(dt + j*src.cols, _mm256_mul_ps(half, _mm256_add_ps(_mm256_loadu_ps(b + j * 8), pv)));
		}
	}
}

void LaplacianSmoothingIIRFilterAVX(Mat& src, Mat& dest, const float sigma)
{
	if (dest.empty())dest.create(src.size(), src.type());
	Mat buff(max(src.cols * 8, src.rows * 8), 1, src.type());

	const __m256i gidx = _mm256_set_epi32(0, src.cols, 2 * src.cols, 3 * src.cols, 4 * src.cols, 5 * src.cols, 6 * src.cols, 7 * src.cols);
	for (int j = 0; j < src.rows; j += 8)
	{
		const __m256 ms = _mm256_set1_ps(sigma);
		const __m256 mis = _mm256_set1_ps(1.f - sigma);
		const __m256 half = _mm256_set1_ps(0.5f);
		float* im = src.ptr<float>(j);
		float* dt = dest.ptr<float>(j);
		float* b = buff.ptr<float>(0);

		const int idx0 = 0;
		const int idx1 = 1 * src.cols;
		const int idx2 = 2 * src.cols;
		const int idx3 = 3 * src.cols;
		const int idx4 = 4 * src.cols;
		const int idx5 = 5 * src.cols;
		const int idx6 = 6 * src.cols;
		const int idx7 = 7 * src.cols;

		__m256 pv = _mm256_i32gather_ps(im, gidx, 4);
		_mm256_store_ps(b, pv);
		dt[idx0] = im[idx0];
		dt[idx1] = im[idx1];
		dt[idx2] = im[idx2];
		dt[idx3] = im[idx3];
		dt[idx4] = im[idx4];
		dt[idx5] = im[idx5];
		dt[idx6] = im[idx6];
		dt[idx7] = im[idx7];
		for (int i = 1; i < src.cols; i++)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i, gidx, 4), _mm256_mul_ps(mis, pv));
			_mm256_store_ps(b + 8 * i, pv);
		}
		__m256 t = pv;
		pv = _mm256_i32gather_ps(im + src.cols - 1, gidx, 4);
		t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));
		dt[idx0 + src.cols - 1] = t.m256_f32[7];
		dt[idx1 + src.cols - 1] = t.m256_f32[6];
		dt[idx2 + src.cols - 1] = t.m256_f32[5];
		dt[idx3 + src.cols - 1] = t.m256_f32[4];
		dt[idx4 + src.cols - 1] = t.m256_f32[3];
		dt[idx5 + src.cols - 1] = t.m256_f32[2];
		dt[idx6 + src.cols - 1] = t.m256_f32[1];
		dt[idx7 + src.cols - 1] = t.m256_f32[0];
		/*for (int i = src.cols - 2; i >= 0; i--)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i, gidx, 4), _mm256_mul_ps(mis, pv));
			__m256 v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * i), pv));
			dt[i + idx0] = v.m256_f32[7];
			dt[i + idx1] = v.m256_f32[6];
			dt[i + idx2] = v.m256_f32[5];
			dt[i + idx3] = v.m256_f32[4];
			dt[i + idx4] = v.m256_f32[3];
			dt[i + idx5] = v.m256_f32[2];
			dt[i + idx6] = v.m256_f32[1];
			dt[i + idx7] = v.m256_f32[0];
		}*/
		
		for (int i = src.cols - 2; i >= 8; i -= 8)
		//for (int i = src.cols - 2; i >= 16; i -= 16)
		//for (int i = src.cols - 2; i >= 0; i--)
		{
			//if (i % 8 == 7)
			//{
			//	_mm_prefetch((const char*)(im + i - 7 + idx0), _MM_HINT_T0);
			//	_mm_prefetch((const char*)(im + i - 7 + idx1), _MM_HINT_T0);
			//	_mm_prefetch((const char*)(im + i - 7 + idx2), _MM_HINT_T0);
			//	_mm_prefetch((const char*)(im + i - 7 + idx3), _MM_HINT_T0);
			//	_mm_prefetch((const char*)(im + i - 7 + idx4), _MM_HINT_T0);
			//	_mm_prefetch((const char*)(im + i - 7 + idx5), _MM_HINT_T0);
			//	_mm_prefetch((const char*)(im + i - 7 + idx6), _MM_HINT_T0);
			//	_mm_prefetch((const char*)(im + i - 7 + idx7), _MM_HINT_T0);
			//}
			//_mm256_i32gather_ps(im + i - 7, gidx, 4);

			__m256 tmpp = _mm256_i32gather_ps(im + i - 7, gidx, 4);
			//__m256 tmpp = _mm256_i32gather_ps(im + i - 15, gidx, 4);

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i, gidx, 4), _mm256_mul_ps(mis, pv));
			__m256 v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * i), pv));
			dt[i + idx0] = v.m256_f32[7];
			dt[i + idx1] = v.m256_f32[6];
			dt[i + idx2] = v.m256_f32[5];
			dt[i + idx3] = v.m256_f32[4];
			dt[i + idx4] = v.m256_f32[3];
			dt[i + idx5] = v.m256_f32[2];
			dt[i + idx6] = v.m256_f32[1];
			dt[i + idx7] = v.m256_f32[0];
			
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 1, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 1)), pv));
			dt[i - 1 + idx0] = v.m256_f32[7];
			dt[i - 1 + idx1] = v.m256_f32[6];
			dt[i - 1 + idx2] = v.m256_f32[5];
			dt[i - 1 + idx3] = v.m256_f32[4];
			dt[i - 1 + idx4] = v.m256_f32[3];
			dt[i - 1 + idx5] = v.m256_f32[2];
			dt[i - 1 + idx6] = v.m256_f32[1];
			dt[i - 1 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 2, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 2)), pv));
			dt[i - 2 + idx0] = v.m256_f32[7];
			dt[i - 2 + idx1] = v.m256_f32[6];
			dt[i - 2 + idx2] = v.m256_f32[5];
			dt[i - 2 + idx3] = v.m256_f32[4];
			dt[i - 2 + idx4] = v.m256_f32[3];
			dt[i - 2 + idx5] = v.m256_f32[2];
			dt[i - 2 + idx6] = v.m256_f32[1];
			dt[i - 2 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 3, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 3)), pv));
			dt[i - 3 + idx0] = v.m256_f32[7];
			dt[i - 3 + idx1] = v.m256_f32[6];
			dt[i - 3 + idx2] = v.m256_f32[5];
			dt[i - 3 + idx3] = v.m256_f32[4];
			dt[i - 3 + idx4] = v.m256_f32[3];
			dt[i - 3 + idx5] = v.m256_f32[2];
			dt[i - 3 + idx6] = v.m256_f32[1];
			dt[i - 3 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 4, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 4)), pv));
			dt[i - 4 + idx0] = v.m256_f32[7];
			dt[i - 4 + idx1] = v.m256_f32[6];
			dt[i - 4 + idx2] = v.m256_f32[5];
			dt[i - 4 + idx3] = v.m256_f32[4];
			dt[i - 4 + idx4] = v.m256_f32[3];
			dt[i - 4 + idx5] = v.m256_f32[2];
			dt[i - 4 + idx6] = v.m256_f32[1];
			dt[i - 4 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 5, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 5)), pv));
			dt[i - 5 + idx0] = v.m256_f32[7];
			dt[i - 5 + idx1] = v.m256_f32[6];
			dt[i - 5 + idx2] = v.m256_f32[5];
			dt[i - 5 + idx3] = v.m256_f32[4];
			dt[i - 5 + idx4] = v.m256_f32[3];
			dt[i - 5 + idx5] = v.m256_f32[2];
			dt[i - 5 + idx6] = v.m256_f32[1];
			dt[i - 5 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 6, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 6)), pv));
			dt[i - 6 + idx0] = v.m256_f32[7];
			dt[i - 6 + idx1] = v.m256_f32[6];
			dt[i - 6 + idx2] = v.m256_f32[5];
			dt[i - 6 + idx3] = v.m256_f32[4];
			dt[i - 6 + idx4] = v.m256_f32[3];
			dt[i - 6 + idx5] = v.m256_f32[2];
			dt[i - 6 + idx6] = v.m256_f32[1];
			dt[i - 6 + idx7] = v.m256_f32[0];

			//pv = _mm256_fmadd_ps(ms, tmpp, _mm256_mul_ps(mis, pv));
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 7, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 7)), pv));
			dt[i - 7 + idx0] = v.m256_f32[7];
			dt[i - 7 + idx1] = v.m256_f32[6];
			dt[i - 7 + idx2] = v.m256_f32[5];
			dt[i - 7 + idx3] = v.m256_f32[4];
			dt[i - 7 + idx4] = v.m256_f32[3];
			dt[i - 7 + idx5] = v.m256_f32[2];
			dt[i - 7 + idx6] = v.m256_f32[1];
			dt[i - 7 + idx7] = v.m256_f32[0];
			

			/*
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 8, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 8)), pv));
			dt[i -8 + idx0] = v.m256_f32[7];
			dt[i -8 + idx1] = v.m256_f32[6];
			dt[i -8 + idx2] = v.m256_f32[5];
			dt[i -8 + idx3] = v.m256_f32[4];
			dt[i -8 + idx4] = v.m256_f32[3];
			dt[i -8 + idx5] = v.m256_f32[2];
			dt[i -8 + idx6] = v.m256_f32[1];
			dt[i -8 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 9, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 9)), pv));
			dt[i - 9 + idx0] = v.m256_f32[7];
			dt[i - 9 + idx1] = v.m256_f32[6];
			dt[i - 9 + idx2] = v.m256_f32[5];
			dt[i - 9 + idx3] = v.m256_f32[4];
			dt[i - 9 + idx4] = v.m256_f32[3];
			dt[i - 9 + idx5] = v.m256_f32[2];
			dt[i - 9 + idx6] = v.m256_f32[1];
			dt[i - 9 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 10, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 10)), pv));
			dt[i - 10 + idx0] = v.m256_f32[7];
			dt[i - 10 + idx1] = v.m256_f32[6];
			dt[i - 10 + idx2] = v.m256_f32[5];
			dt[i - 10 + idx3] = v.m256_f32[4];
			dt[i - 10 + idx4] = v.m256_f32[3];
			dt[i - 10 + idx5] = v.m256_f32[2];
			dt[i - 10 + idx6] = v.m256_f32[1];
			dt[i - 10 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 11, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 11)), pv));
			dt[i - 11 + idx0] = v.m256_f32[7];
			dt[i - 11 + idx1] = v.m256_f32[6];
			dt[i - 11 + idx2] = v.m256_f32[5];
			dt[i - 11 + idx3] = v.m256_f32[4];
			dt[i - 11 + idx4] = v.m256_f32[3];
			dt[i - 11 + idx5] = v.m256_f32[2];
			dt[i - 11 + idx6] = v.m256_f32[1];
			dt[i - 11 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 12, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 12)), pv));
			dt[i - 12+ idx0] = v.m256_f32[7];
			dt[i - 12+ idx1] = v.m256_f32[6];
			dt[i - 12+ idx2] = v.m256_f32[5];
			dt[i - 12+ idx3] = v.m256_f32[4];
			dt[i - 12+ idx4] = v.m256_f32[3];
			dt[i - 12+ idx5] = v.m256_f32[2];
			dt[i - 12+ idx6] = v.m256_f32[1];
			dt[i - 12+ idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 13, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 13)), pv));
			dt[i - 13 + idx0] = v.m256_f32[7];
			dt[i - 13 + idx1] = v.m256_f32[6];
			dt[i - 13 + idx2] = v.m256_f32[5];
			dt[i - 13 + idx3] = v.m256_f32[4];
			dt[i - 13 + idx4] = v.m256_f32[3];
			dt[i - 13 + idx5] = v.m256_f32[2];
			dt[i - 13 + idx6] = v.m256_f32[1];
			dt[i - 13 + idx7] = v.m256_f32[0];

			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 14, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 14)), pv));
			dt[i - 14 + idx0] = v.m256_f32[7];
			dt[i - 14 + idx1] = v.m256_f32[6];
			dt[i - 14 + idx2] = v.m256_f32[5];
			dt[i - 14 + idx3] = v.m256_f32[4];
			dt[i - 14 + idx4] = v.m256_f32[3];
			dt[i - 14 + idx5] = v.m256_f32[2];
			dt[i - 14 + idx6] = v.m256_f32[1];
			dt[i - 14 + idx7] = v.m256_f32[0];

			//pv = _mm256_fmadd_ps(ms, tmpp, _mm256_mul_ps(mis, pv));
			pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(im + i - 15, gidx, 4), _mm256_mul_ps(mis, pv));
			v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(b + 8 * (i - 15)), pv));
			dt[i - 15 + idx0] = v.m256_f32[7];
			dt[i - 15 + idx1] = v.m256_f32[6];
			dt[i - 15 + idx2] = v.m256_f32[5];
			dt[i - 15 + idx3] = v.m256_f32[4];
			dt[i - 15 + idx4] = v.m256_f32[3];
			dt[i - 15 + idx5] = v.m256_f32[2];
			dt[i - 15 + idx6] = v.m256_f32[1];
			dt[i - 15 + idx7] = v.m256_f32[0];
			*/
		}
	}

	for (int i = 0; i < src.cols; i += 8)
	{
		const __m256 ms = _mm256_set1_ps(sigma);
		const __m256 mis = _mm256_set1_ps(1.f - sigma);
		const __m256 half = _mm256_set1_ps(0.5f);
		float* im = dest.ptr<float>(0) + i;
		float* dt = dest.ptr<float>(0) + i;
		float* b = buff.ptr<float>(0);

		__m256 pv = _mm256_load_ps(im);
		_mm256_storeu_ps(b, pv);
		for (int j = 1; j < src.rows; j++)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_loadu_ps(im + j*src.cols), _mm256_mul_ps(mis, pv));
			_mm256_store_ps(b + j * 8, pv);
		}

		__m256 t = pv;
		pv = _mm256_loadu_ps(im + src.cols*(src.rows - 1));
		t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));
		_mm256_storeu_ps(dt + (src.rows - 1)*src.cols, t);

		for (int j = src.rows - 2; j >= 0; j--)
		{
			pv = _mm256_fmadd_ps(ms, _mm256_loadu_ps(im + j*src.cols), _mm256_mul_ps(mis, pv));
			_mm256_storeu_ps(dt + j*src.cols, _mm256_mul_ps(half, _mm256_add_ps(_mm256_loadu_ps(b + j * 8), pv)));
		}
	}
}

void LaplacianSmoothingIIRFilter(Mat& src, Mat& dest, const double sigma_, int opt=VECTOR_AVX)
{
	if (opt == VECTOR_WITHOUT) LaplacianSmoothingIIRFilterBase(src, dest, sigma_);
	//else if (opt == VECTOR_AVX)LaplacianSmoothingIIRFilterAVX(src, dest, (float)sigma_);
	else if (opt == VECTOR_AVX)LaplacianSmoothingIIRFilterAVXUnroall8(src, dest, (float)sigma_);
}

class LaplacianSmoothingIIR
{
	Mat hfilter;
public:
	void filter(Mat& src, Mat& dest, float sigma)
	{
		if (dest.empty()) dest.create(src.size(), src.type());	
		if (src.cols % 1024 == 0)
		{
			if(hfilter.empty())hfilter.create(Size(src.cols + 8, src.rows), src.type());
		}
		else
		{
			hfilter = dest;
		}

		const int hstep = hfilter.cols;
		Mat buff(max(src.cols * 16, src.rows * 16), 1, src.type());
		const int idx0 = 0;
		const int idx1 = 1 * src.cols;
		const int idx2 = 2 * src.cols;
		const int idx3 = 3 * src.cols;
		const int idx4 = 4 * src.cols;
		const int idx5 = 5 * src.cols;
		const int idx6 = 6 * src.cols;
		const int idx7 = 7 * src.cols;
		const int hidx0 = 0;
		const int hidx1 = 1 * hstep;
		const int hidx2 = 2 * hstep;
		const int hidx3 = 3 * hstep;
		const int hidx4 = 4 * hstep;
		const int hidx5 = 5 * hstep;
		const int hidx6 = 6 * hstep;
		const int hidx7 = 7 * hstep;
		float* b = buff.ptr<float>(0);
		for (int j = 0; j < src.rows; j += 8)
		{
			const __m256 ms = _mm256_set1_ps(sigma);
			const __m256 mis = _mm256_set1_ps(1.f - sigma);
			const __m256 half = _mm256_set1_ps(0.5f);
			const __m256i gidx = _mm256_set_epi32(0, src.cols, 2 * src.cols, 3 * src.cols, 4 * src.cols, 5 * src.cols, 6 * src.cols, 7 * src.cols);
			float* im = src.ptr<float>(j);
			float* dt = hfilter.ptr<float>(j);
			
			__m256 pv = _mm256_i32gather_ps(im, gidx, 4);
			_mm256_store_ps(b, pv);
			dt[hidx0] = im[idx0];
			dt[hidx1] = im[idx1];
			dt[hidx2] = im[idx2];
			dt[hidx3] = im[idx3];
			dt[hidx4] = im[idx4];
			dt[hidx5] = im[idx5];
			dt[hidx6] = im[idx6];
			dt[hidx7] = im[idx7];
			{
				float* ip = im + 1;
				float* bp = b + 8;
				for (int i = 1; i < src.cols; i++)
				{
					pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(ip, gidx, 4), _mm256_mul_ps(mis, pv));
					_mm256_store_ps(bp, pv);
					ip++;
					bp += 8;
				}
			}
			__m256 t = pv;
			pv = _mm256_i32gather_ps(im + src.cols - 1, gidx, 4);
			t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));
			dt[hidx0 + src.cols - 1] = t.m256_f32[7];
			dt[hidx1 + src.cols - 1] = t.m256_f32[6];
			dt[hidx2 + src.cols - 1] = t.m256_f32[5];
			dt[hidx3 + src.cols - 1] = t.m256_f32[4];
			dt[hidx4 + src.cols - 1] = t.m256_f32[3];
			dt[hidx5 + src.cols - 1] = t.m256_f32[2];
			dt[hidx6 + src.cols - 1] = t.m256_f32[1];
			dt[hidx7 + src.cols - 1] = t.m256_f32[0];

			{
				float* ip = im + src.cols - 2;
				float* bp = b + 8*(src.cols - 2);
				for (int i = src.cols - 2; i >= 0; i--)
				{
					//__m256 tmpp = _mm256_i32gather_ps(im + i - 7, gidx, 4);
					pv = _mm256_fmadd_ps(ms, _mm256_i32gather_ps(ip, gidx, 4), _mm256_mul_ps(mis, pv));
					__m256 v = _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(bp), pv));
					dt[i + hidx0] = v.m256_f32[7];
					dt[i + hidx1] = v.m256_f32[6];
					dt[i + hidx2] = v.m256_f32[5];
					dt[i + hidx3] = v.m256_f32[4];
					dt[i + hidx4] = v.m256_f32[3];
					dt[i + hidx5] = v.m256_f32[2];
					dt[i + hidx6] = v.m256_f32[1];
					dt[i + hidx7] = v.m256_f32[0];
					ip--;
					bp -= 8;
				}
			}
		}

		for (int i = 0; i < src.cols; i += 16)
		{
			const __m256 ms = _mm256_set1_ps(sigma);
			const __m256 mis = _mm256_set1_ps(1.f - sigma);
			const __m256 half = _mm256_set1_ps(0.5f);

			float* im = hfilter.ptr<float>(0) + i;
			float* dt = dest.ptr<float>(0) + i;
			float* b = buff.ptr<float>(0);

			__m256 pv = _mm256_loadu_ps(im);
			__m256 pv1 = _mm256_load_ps(im+8);
			_mm256_store_ps(b, pv);
			_mm256_store_ps(b + 8, pv1);
			{
				float* ip = im + hstep;
				float* bp = b + 16;
				for (int j = 1; j < src.rows; j++)
				{
					pv = _mm256_fmadd_ps(ms, _mm256_loadu_ps(ip), _mm256_mul_ps(mis, pv));
					_mm256_store_ps(bp, pv);
					pv1 = _mm256_fmadd_ps(ms, _mm256_loadu_ps(ip + 8), _mm256_mul_ps(mis, pv1));
					_mm256_store_ps(bp + 8, pv1);
					ip += hstep;
					bp += 16;
				}
			}

			__m256 t = pv;
			pv = _mm256_loadu_ps(im + hstep*(src.rows - 1));
			t = _mm256_mul_ps(half, _mm256_add_ps(t, pv));
			_mm256_store_ps(dt + (src.rows - 1)*src.cols, t);

			t = pv1;
			pv1 = _mm256_loadu_ps(im + 8+ hstep*(src.rows - 1));
			t = _mm256_mul_ps(half, _mm256_add_ps(t, pv1));
			_mm256_store_ps(dt + 8 + (src.rows - 1)*src.cols, t);

			{
				float* ip = im + hstep*(src.rows - 2);
				float* bp = b + 16*(src.rows - 2);
				float* dtp = dt + src.cols*(src.rows - 2);
				for (int j = src.rows - 2; j >= 0; j--)
				{
					pv = _mm256_fmadd_ps(ms, _mm256_load_ps(ip), _mm256_mul_ps(mis, pv));
					_mm256_storeu_ps(dtp, _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(bp), pv)));
					pv1 = _mm256_fmadd_ps(ms, _mm256_load_ps(ip + 8), _mm256_mul_ps(mis, pv1));
					_mm256_storeu_ps(dtp + 8, _mm256_mul_ps(half, _mm256_add_ps(_mm256_load_ps(bp + 8), pv1)));

					ip -= hstep;
					bp -= 16;
					dtp -= src.cols;
				}
			}
		}
	}
};