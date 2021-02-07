/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "videoSource.h"
#include "videoOutput.h"
#include "loadImage.h"

#include "cudaOverlay.h"
#include "cudaMappedMemory.h"

#include "segNet.h"

#include <signal.h>


#ifdef HEADLESS
	#define IS_HEADLESS() "headless"             // run without display
	#define DEFAULT_VISUALIZATION "overlay"      // output overlay only
#else
	#define IS_HEADLESS() (const char*)NULL      // use display (if attached)
	#define DEFAULT_VISUALIZATION "overlay|mask" // output overlay + mask
#endif


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: segnet [--help] [--network NETWORK] ...\n");
	printf("              input_URI [output_URI]\n\n");
	printf("Segment and classify a video/image stream using a semantic segmentation DNN.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s\n", segNet::Usage());
	printf("%s\n", videoSource::Usage());
	printf("%s\n", videoOutput::Usage());
	printf("%s\n", Log::Usage());

	return 0;
}


//
// segmentation buffers
//
typedef uchar3 pixelType;		// this can be uchar3, uchar4, float3, float4

/* BACKGROUND_MATTING_V2 */
pixelType* imgBgrInput    = NULL;	// BACKGROUND_MATTING_V2:bgr
pixelType* imgMaskOutput = NULL;	// BACKGROUND_MATTING_V2:BinaryMask
pixelType* imgBlendOutput = NULL;	// BACKGROUND_MATTING_V2:BlendingImage

/* BACKGROUND_MATTING_V2 */
int2 bgrinputSize;		// BACKGROUND_MATTING_V2:bgr
int2 maskoutputSize;	// BACKGROUND_MATTING_V2:BinaryMask
int2 blendoutputSize;	// BACKGROUND_MATTING_V2:BlendingImage

/* SEGNET */
pixelType* imgMask      = NULL;	// color of each segmentation class
pixelType* imgOverlay   = NULL;	// input + alpha-blended mask
pixelType* imgComposite = NULL;	// overlay with mask next to it
pixelType* imgOutput    = NULL;	// reference to one of the above three

/* SEGNET */
int2 maskSize;
int2 overlaySize;
int2 compositeSize;
int2 outputSize;
segNet::FilterMode filterMode;
uint32_t visualizationFlags;
const char* ignoreClass;

// allocate BACKGROUND_MATTING_V2 buffers
bool allocBackGroundMattingV2Buffers( int width, int height)
{
	// free previous buffers if they exit
	CUDA_FREE_HOST(imgMaskOutput);
	CUDA_FREE_HOST(imgBlendOutput);

	// allocate output BlendingImage image
	maskoutputSize = make_int2(width, height);

	if( !cudaAllocMapped(&imgMaskOutput, maskoutputSize) )
	{
		LogError("BACKGROUND_MATTING_V2:  failed to allocate CUDA memory for output BinaryMask image (%ux%u)\n", width, height);
		return false;
	}

	// allocate output BlendingImage image
	blendoutputSize = make_int2(width, height);

	if( !cudaAllocMapped(&imgBlendOutput, blendoutputSize) )
	{
		LogError("BACKGROUND_MATTING_V2:  failed to allocate CUDA memory for output BlendingImage image (%ux%u)\n", width, height);
		return false;
	}

	return true;
}

// allocate mask/overlay output buffers
bool allocBuffers( int width, int height, uint32_t flags )
{
	// check if the buffers were already allocated for this size
	if( imgOverlay != NULL && width == overlaySize.x && height == overlaySize.y )
		return true;

	// free previous buffers if they exit
	CUDA_FREE_HOST(imgMask);
	CUDA_FREE_HOST(imgOverlay);
	CUDA_FREE_HOST(imgComposite);

	// allocate overlay image
	overlaySize = make_int2(width, height);
	
	if( flags & segNet::VISUALIZE_OVERLAY )
	{
		if( !cudaAllocMapped(&imgOverlay, overlaySize) )
		{
			LogError("segnet:  failed to allocate CUDA memory for overlay image (%ux%u)\n", width, height);
			return false;
		}

		imgOutput = imgOverlay;
		outputSize = overlaySize;
	}

	// allocate mask image (half the size, unless it's the only output)
	if( flags & segNet::VISUALIZE_MASK )
	{
		maskSize = (flags & segNet::VISUALIZE_OVERLAY) ? make_int2(width/2, height/2) : overlaySize;

		if( !cudaAllocMapped(&imgMask, maskSize) )
		{
			LogError("segnet:  failed to allocate CUDA memory for mask image\n");
			return false;
		}

		imgOutput = imgMask;
		outputSize = maskSize;
	}

	// allocate composite image if both overlay and mask are used
	if( (flags & segNet::VISUALIZE_OVERLAY) && (flags & segNet::VISUALIZE_MASK) )
	{
		compositeSize = make_int2(overlaySize.x + maskSize.x, overlaySize.y);

		if( !cudaAllocMapped(&imgComposite, compositeSize) )
		{
			LogError("segnet:  failed to allocate CUDA memory for composite image\n");
			return false;
		}

		imgOutput = imgComposite;
		outputSize = compositeSize;
	}

	return true;
}


int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv, IS_HEADLESS());

	if( cmdLine.GetFlag("help") )
		return usage();


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * create input stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("segnet:  failed to create input stream\n");
		return 0;
	}


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
		LogError("segnet:  failed to create output stream\n");	
	

	/*
	 * create segmentation network
	 */
	segNet* net = segNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("segnet:  failed to initialize segNet\n");
		return 0;
	}

	// get the desired NetworkType type
	const segNet::NetworkType networkType = segNet::NetworkTypeFromStr(cmdLine.GetString("network", "fcn-resnet18-voc-320x320"));

	if( networkType == segNet::BACKGROUND_MATTING_V2 )
	{
		printf(LOG_TRT "networkType (%d) \n", networkType);

		CUDA_FREE_HOST(imgBgrInput);

		// allocate input bgr image
		bgrinputSize = make_int2(1920, 1080);

		if( !cudaAllocMapped(&imgBgrInput, bgrinputSize) )
		{
			LogError("BACKGROUND_MATTING_V2:  failed to allocate CUDA memory for input bgr image\n");
			return false;
		}
/*
		//[DEBUG] load bgr image from file.
		if( !loadImage("test_img_bg.png", (void**)&imgBgrInput, &bgrinputSize.x, &bgrinputSize.y, IMAGE_RGB8) )
		{
			printf("segnet:  failed to load image '%s'\n", "test_img_bg.png");
			return 0;
		}
*/
	}
	else
	{
		// set alpha blending value for classes that don't explicitly already have an alpha	
		net->SetOverlayAlpha(cmdLine.GetFloat("alpha", 150.0f));

		// get the desired overlay/mask filtering mode
		filterMode = segNet::FilterModeFromStr(cmdLine.GetString("filter-mode", "linear"));

		// get the visualization flags
		visualizationFlags = segNet::VisualizationFlagsFromStr(cmdLine.GetString("visualize", DEFAULT_VISUALIZATION));

		// get the object class to ignore (if any)
		ignoreClass = cmdLine.GetString("ignore-class", "void");
	}



	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture next image image
		pixelType* imgInput = NULL;

		if( !input->Capture(&imgInput, 1000) )
		{
			// check for EOS
			if( !input->IsStreaming() )
				break; 

			LogError("segnet:  failed to capture video frame\n");
			continue;
		}
		
		printf(LOG_TRT "Capture (Width,Height) (%d,%d) \n", input->GetWidth(),input->GetHeight());

		if( networkType == segNet::BACKGROUND_MATTING_V2 )
		{
			if(imgBgrInput->x==0)
			{
				// Take a trigger and copy imgBgrInput from imgInput.
				CUDA(cudaMemcpy(imgBgrInput, imgInput, imageFormatSize(IMAGE_RGB8, input->GetWidth(), input->GetHeight()), cudaMemcpyDeviceToDevice));
			}
/*
			printf("imgInput:%d \n",  imgInput->x);
			printf("imgBgrInput:%d \n",  imgBgrInput->x);
			printf("imgInput[100]:%d \n",  imgInput[100].x);
			printf("imgBgrInput[100]:%d \n",  imgBgrInput[100].x);
*/
		}

		/*--------------*/
		/* allocBuffers */
		/*--------------*/

		if( networkType == segNet::BACKGROUND_MATTING_V2 )
		{
			// allocate backgroundmattingv2 buffers for this size frame
			if( !allocBackGroundMattingV2Buffers(input->GetWidth(), input->GetHeight()) )
			{
				LogError("BackGroundMattingV2:  failed to allocate buffers\n");
				continue;
			};
		}
		else
		{
			// allocate buffers for this size frame
			if( !allocBuffers(input->GetWidth(), input->GetHeight(), visualizationFlags) )
			{
				LogError("segnet:  failed to allocate buffers\n");
				continue;
			}
		}

		/*---------*/
		/* Process */
		/*---------*/

		if( networkType == segNet::BACKGROUND_MATTING_V2 )
		{
			// process the segmentation network
			if( !net->Process(imgInput, imgBgrInput, input->GetWidth(), input->GetHeight()) )
			{
				LogError("BACKGROUND_MATTING_V2:  failed to process segmentation\n");
				continue;
			}
		}
		else
		{
			// process the segmentation network
			if( !net->Process(imgInput, input->GetWidth(), input->GetHeight(), ignoreClass) )
			{
				LogError("segnet:  failed to process segmentation\n");
				continue;
			}
		}

		/*---------------*/
		/* Visualization */
		/*---------------*/
		
		if( networkType == segNet::BACKGROUND_MATTING_V2 )
		{
			if( !net->BinaryMask(imgMaskOutput, maskoutputSize.x, maskoutputSize.y) )
			{
				LogError("segnet:-console:  failed to process BinaryMask.\n");
				continue;
			}
			
			if( !net->BlendingImage(imgBlendOutput, blendoutputSize.x, blendoutputSize.y) )
			{
				LogError("segnet:-console:  failed to process BlendingImage.\n");
				continue;
			}
		}
		else
		{
			// generate overlay
			if( visualizationFlags & segNet::VISUALIZE_OVERLAY )
			{
				if( !net->Overlay(imgOverlay, overlaySize.x, overlaySize.y, filterMode) )
				{
					LogError("segnet:  failed to process segmentation overlay.\n");
					continue;
				}
			}

			// generate mask
			if( visualizationFlags & segNet::VISUALIZE_MASK )
			{
				if( !net->Mask(imgMask, maskSize.x, maskSize.y, filterMode) )
				{
					LogError("segnet:-console:  failed to process segmentation mask.\n");
					continue;
				}
			}

			// generate composite
			if( (visualizationFlags & segNet::VISUALIZE_OVERLAY) && (visualizationFlags & segNet::VISUALIZE_MASK) )
			{
				CUDA(cudaOverlay(imgOverlay, overlaySize, imgComposite, compositeSize, 0, 0));
				CUDA(cudaOverlay(imgMask, maskSize, imgComposite, compositeSize, overlaySize.x, 0));
			}
		}

		// render outputs
		if( output != NULL )
		{
			if( networkType == segNet::BACKGROUND_MATTING_V2 )
			{
				//output->Render(imgBgrInput, bgrinputSize.x, bgrinputSize.y);
				//output->Render(imgInput, bgrinputSize.x, bgrinputSize.y);
				//output->Render(imgMaskOutput, maskoutputSize.x, maskoutputSize.y);
				output->Render(imgBlendOutput, blendoutputSize.x, blendoutputSize.y);
			}
			else
			{
				output->Render(imgOutput, outputSize.x, outputSize.y);
				//output->Render(imgOverlay, overlaySize.x, overlaySize.y);
				//output->Render(imgMask, maskSize.x, maskSize.y);
			}

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->GetNetworkName(), net->GetNetworkFPS());
			output->SetStatus(str);

			// check if the user quit
			if( !output->IsStreaming() )
				signal_recieved = true;
		}

		// wait for the GPU to finish		
		CUDA(cudaDeviceSynchronize());

		// print out timing info
		net->PrintProfilerTimes();
	}
	

	/*
	 * destroy resources
	 */
	LogVerbose("segnet:  shutting down...\n");

	SAFE_DELETE(input);
	SAFE_DELETE(output);

	if( networkType == segNet::BACKGROUND_MATTING_V2 )
	{
		CUDA_FREE_HOST(imgBgrInput);
		CUDA_FREE_HOST(imgMaskOutput);
		CUDA_FREE_HOST(imgBlendOutput);
	}
	else
	{
		SAFE_DELETE(net);
		CUDA_FREE_HOST(imgMask);
		CUDA_FREE_HOST(imgOverlay);
		CUDA_FREE_HOST(imgComposite);
	}


	LogVerbose("segnet:  shutdown complete.\n");
	return 0;
}

