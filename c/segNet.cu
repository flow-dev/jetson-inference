/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "cudaUtility.h"
#include "cudaVector.h"
#include "segNet.h"


// gpuSegOverlay
template<typename T, bool filter_linear, bool mask_only>
__global__ void gpuSegOverlay( T* input, const int in_width, const int in_height,
						 T* output, const int out_width, const int out_height,
						 float4* class_colors, uint8_t* scores, const int2 scores_dim )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= out_width || y >= out_height )
		return;

	const float px = float(x) / float(out_width);
	const float py = float(y) / float(out_height);

	#define LOOKUP_CLASS_MAP(ix, iy)	scores[iy * scores_dim.x + ix]

	// point or linear filtering mode
	if( !filter_linear )
	{
		// calculate coordinates in scores cell
		const float cx = px * float(scores_dim.x);	
		const float cy = py * float(scores_dim.y);

		const int x1 = int(cx);
		const int y1 = int(cy);

		// get the class ID of this cell
		const uint8_t classIdx = LOOKUP_CLASS_MAP(x1, y1);

		// find the color of this class
		const float4 classColor = class_colors[classIdx];

		// output the pixel
		if( mask_only )
		{
			// only draw the segmentation mask
			output[y * out_width + x] = make_vec<T>(classColor.x, classColor.y, classColor.z, 255);
		}
		else
		{
			// alpha blend with input image
			const int x_in = px * float(in_width);
			const int y_in = py * float(in_height);

			const T px_in = input[y_in * in_width + x_in];

			const float alph = classColor.w / 255.0f;
			const float inva = 1.0f - alph;

			output[y * out_width + x] = make_vec<T>(
				alph * classColor.x + inva * px_in.x,
				alph * classColor.y + inva * px_in.y,
				alph * classColor.z + inva * px_in.z,
				255.0f);
		}
	}
	else
	{
		// calculate coordinates in scores cell
		const float bx = (px * float(scores_dim.x)) - 0.5f;
		const float by = (py * float(scores_dim.y)) - 0.5f;

		const float cx = bx < 0.0f ? 0.0f : bx;
		const float cy = by < 0.0f ? 0.0f : by;

		const int x1 = int(cx);
		const int y1 = int(cy);
			
		const int x2 = x1 >= scores_dim.x - 1 ? x1 : x1 + 1;	// bounds check
		const int y2 = y1 >= scores_dim.y - 1 ? y1 : y1 + 1;
		
		const uchar4 classIdx = make_uchar4(LOOKUP_CLASS_MAP(x1, y1),
									 LOOKUP_CLASS_MAP(x2, y1),
									 LOOKUP_CLASS_MAP(x2, y2),
									 LOOKUP_CLASS_MAP(x1, y2));

		const float4 cc[] = { class_colors[classIdx.x],
						  class_colors[classIdx.y],
						  class_colors[classIdx.z],
						  class_colors[classIdx.w] };

		// compute bilinear weights
		const float x1d = cx - float(x1);
		const float y1d = cy - float(y1);

		const float x1f = 1.0f - x1d;
		const float y1f = 1.0f - y1d;

		const float x2f = 1.0f - x1f;
		const float y2f = 1.0f - y1f;

		const float x1y1f = x1f * y1f;
		const float x1y2f = x1f * y2f;
		const float x2y1f = x2f * y1f;
		const float x2y2f = x2f * y2f;

		const float4 classColor = make_float4(
			cc[0].x * x1y1f + cc[1].x * x2y1f + cc[2].x * x2y2f + cc[3].x * x1y2f,
			cc[0].y * x1y1f + cc[1].y * x2y1f + cc[2].y * x2y2f + cc[3].y * x1y2f,
			cc[0].z * x1y1f + cc[1].z * x2y1f + cc[2].z * x2y2f + cc[3].z * x1y2f,
			cc[0].w * x1y1f + cc[1].w * x2y1f + cc[2].w * x2y2f + cc[3].w * x1y2f );

		// output the pixel
		if( mask_only )
		{
			// only draw the segmentation mask
			output[y * out_width + x] = make_vec<T>(classColor.x, classColor.y, classColor.z, 255);
		}
		else
		{
			// alpha blend with input image
			const int x_in = px * float(in_width);
			const int y_in = py * float(in_height);

			const T px_in = input[y_in * in_width + x_in];

			const float alph = classColor.w / 255.0f;
			const float inva = 1.0f - alph;

			output[y * out_width + x] = make_vec<T>(
				alph * classColor.x + inva * px_in.x,
				alph * classColor.y + inva * px_in.y,
				alph * classColor.z + inva * px_in.z,
				255.0f);
		}

	}
}

// cudaSegOverlay
cudaError_t cudaSegOverlay( void* input, uint32_t in_width, uint32_t in_height,
				        void* output, uint32_t out_width, uint32_t out_height, imageFormat format,
					   float4* class_colors, uint8_t* scores, const int2& scores_dim,
					   bool filter_linear, bool mask_only, cudaStream_t stream )
{
	if( !output )
		return cudaErrorInvalidDevicePointer;

	if( out_width == 0 || out_height == 0 )
		return cudaErrorInvalidValue;

	if( !imageFormatIsRGB(format) )
	{
		LogError(LOG_TRT "segNet -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_TRT "          supported formats are:\n");
		LogError(LOG_TRT "              * rgb8\n");		
		LogError(LOG_TRT "              * rgba8\n");		
		LogError(LOG_TRT "              * rgb32f\n");		
		LogError(LOG_TRT "              * rgba32f\n");

		return cudaErrorInvalidValue;
	}

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(out_width,blockDim.x), iDivUp(out_height,blockDim.y));

	#define LAUNCH_OVERLAY_KERNEL(type, filter, mask) gpuSegOverlay<type, filter, mask><<<gridDim, blockDim, 0, stream>>>((type*)input, in_width, in_height, (type*)output, out_width, out_height, class_colors, scores, scores_dim)
	
	#define LAUNCH_OVERLAY(filter, mask) 				\
		if( format == IMAGE_RGB8 ) {					\
			LAUNCH_OVERLAY_KERNEL(uchar3, filter, mask);	\
		}										\
		else if( format == IMAGE_RGBA8 ) {				\
			LAUNCH_OVERLAY_KERNEL(uchar4, filter, mask);	\
		}										\
		else if( format == IMAGE_RGB32F ) {			\
			LAUNCH_OVERLAY_KERNEL(float3, filter, mask);	\
		}										\
		else if( format == IMAGE_RGBA32F )	{			\
			LAUNCH_OVERLAY_KERNEL(float4, filter, mask); \
		}										

	if( filter_linear )
	{
		if( mask_only )
		{
			LAUNCH_OVERLAY(true, true)
		}
		else
		{
			LAUNCH_OVERLAY(true, false)
		}
	}
	else
	{
		if( mask_only )
		{
			LAUNCH_OVERLAY(false, true)
		}
		else
		{
			LAUNCH_OVERLAY(false, false)
		}
	}

	return CUDA(cudaGetLastError());
}


/**
 * CUDA device function for alpha blending two pixels.
 * The alpha blending of the `fgr` and `bgr` pixels is
 * computed by the equation: `fgr * alph + bgr * (1.0 - alph)`
 *
 * @note cudaBlendImage() is for use inside of other CUDA kernels.
 * @ingroup cuda
 */
 __device__ inline float3 cudaBlendImage( const float3& fgr, const float3& bgr, const float& alph )
 {
	 const float inva = 1.0f - alph;
 
	 return make_float3(alph * fgr.x + inva * bgr.x,
					 alph * fgr.y + inva * bgr.y,
					 alph * fgr.z + inva * bgr.z);
 }


// clip float to [min,max]
static inline __device__ float clip( const float x, float min, float max )
{
	return x > max ? max : x < min ? min : x;
}

// clip vector to [min,max]
static inline __device__ uchar3 clip( const float3& px, int min, int max )
{
	return make_uchar3(clip(px.x, min, max),
					   clip(px.y, min, max),
					   clip(px.z, min, max));
}

 // TensorToRGB
 template<typename T_in, typename T_out>
__global__ void TensorToRGB( T_in* srcImage, T_out* dstImage, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = width * height;
	
	if( x >= width || y >= height )
		return;

	const int dx = ((float)x * 1.0f);
	const int dy = ((float)y * 1.0f);

	const uchar3 rgb = clip(make_float3(srcImage[n * 0 + dy * width + dx] * 255.0f,
										srcImage[n * 1 + dy * width + dx] * 255.0f,
										srcImage[n * 2 + dy * width + dx] * 255.0f), 0, 255);

	dstImage[y * width + x] = rgb;
}


template<typename T_in, typename T_out> 
static cudaError_t launchTensorToRGB(  T_in* srcDev, T_out* dstDev, size_t width, size_t height )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	TensorToRGB<T_in, T_out><<<gridDim, blockDim>>>( srcDev, dstDev, width, height );

	return CUDA(cudaGetLastError());
}

// cudaTensor32ToRGB8 (float-> uchar3)
cudaError_t cudaTensor32ToRGB8( float* srcDev, uchar3* dstDev, size_t width, size_t height)
{
	return launchTensorToRGB<float, uchar3>(srcDev, dstDev, width, height);
}


// BlendRGBA32ToRGB8
template<typename T_in_src, typename T_in_bgr, typename T_out> 
__global__ void BlendRGBA32ToRGB8( T_in_src* srcImage, T_in_bgr* alphImage, T_out* dstImage, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = width * height;
	
	if( x >= width || y >= height )
		return;

	const int dx = ((float)x * 1.0f);
	const int dy = ((float)y * 1.0f);

	const uchar3 rgb = clip(make_float3((srcImage[n * 0 + dy * width + dx] * 255.0f * alphImage[dy * width + dx]) + ((1.0f - alphImage[dy * width + dx]) * 120.0f),
										(srcImage[n * 1 + dy * width + dx] * 255.0f * alphImage[dy * width + dx]) + ((1.0f - alphImage[dy * width + dx]) * 255.0f),
										(srcImage[n * 2 + dy * width + dx] * 255.0f * alphImage[dy * width + dx]) + ((1.0f - alphImage[dy * width + dx]) * 155.0f)), 0, 255);

	dstImage[y * width + x] = rgb;
}

template<typename T_in_src, typename T_in_bgr, typename T_out> 
static cudaError_t launchBlendRGBA32ToRGB8(  T_in_src* srcDev, T_in_bgr* alphDev, T_out* dstDev, size_t width, size_t height )
{
	if( !srcDev || !alphDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	BlendRGBA32ToRGB8<T_in_src, T_in_bgr, T_out><<<gridDim, blockDim>>>( srcDev, alphDev, dstDev, width, height );

	return CUDA(cudaGetLastError());
}

// cudaBlendRGBA32ToRGB8 (float(rgb-3ch),float(a-1ch)-> uchar3)
cudaError_t cudaBlendRGBA32ToRGB8( float* srcDev, float* alphDev, uchar3* dstDev, size_t width, size_t height)
{
	return launchBlendRGBA32ToRGB8<float, float, uchar3>(srcDev, alphDev, dstDev, width, height);
}
