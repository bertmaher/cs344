// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words,
// we multiply each weight with the pixel underneath it. Finally, we add up all
// of the multiplied numbers and assign that value to our output for the current
// pixel. We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the
// outputChannel.

// Here is an example of computing a blur, using a weighted average, for a
// single pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its
// width. We refer to the array of weights as a filter, and we refer to its
// width with the variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called
// checkCudaErrors. You should wrap your allocation and copying statements like
// we've done in the code we're supplying you. Here is an example of the unsafe
// way to allocate memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows *
// numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful
// for catching mistakes. If you write code the unsafe way and you make a
// mistake, then any subsequent kernels won't compute anything, and it will be
// hard to figure out why. Writing code the safe way will inform you as soon as
// you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "timer.h"
#include "utils.h"
#include <algorithm>

/**
 * Launch with 32 threads per block.  Each block will handle a 32x32
 * tile of the output.  Each thread will handle a vertical strip of 32
 * pixels within that block.
 */
__global__ void gaussian_blur_fucked(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  int filterWidthSq = filterWidth * filterWidth;
  constexpr int tileSize = 32; // = blockDim.x
  
  extern __shared__ float s_shmem[];
  float* s_filter = s_shmem;
  float* s_tile = s_shmem + filterWidthSq;
  
  // Fetch filter into shared memory.
  for (int i = threadIdx.x; i < filterWidthSq; i += blockDim.x) {
    s_filter[i] = filter[i];
  }

  // Fetch input tile into shared memory.  The tile is 32x32 +
  // filterWidth/2 "halo" rows.  We apply clamping on the inputs here.
  int tileColStart = blockIdx.x * tileSize - filterWidth / 2;
  int tileRowStart = blockIdx.y * tileSize - filterWidth / 2;

  int tileRows = tileSize + filterWidth - 1;
  int tileCols = tileSize + filterWidth - 1;
  for (int i = 0; i < tileRows; i++) {
    for (int j = threadIdx.x; j < tileCols; j += blockDim.x) {
      int row = min(max(i + tileRowStart, 0), numRows - 1);
      int col = min(max(j + tileColStart, 0), numCols - 1);
      s_tile[i * tileCols + j] = inputChannel[(i + tileColStart) * numCols + (j + tileRowStart)];
    }
  }

  __syncthreads();
  
  for (int i = 0; i < tileSize; i++) {
    float res = 0.0f;
    for (int fr = 0; fr < filterWidth; fr++) {
      for (int fc = 0; fc < filterWidth; fc++) {
	res += s_filter[fr * filterWidth + fc] * s_tile[(i + fr) * tileCols + (threadIdx.x + fc)];
      }
    }
    outputChannel[(blockIdx.y * tileSize + i) * numCols + (blockIdx.x * tileSize + threadIdx.x)] = res;
  }
}

__global__ void gaussian_blur_0(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  constexpr int tileSize = 32;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (c >= numCols) {
    return;
  }
  
  for (int rt = 0; rt < tileSize; rt++) {
    int r = rt + blockIdx.y * tileSize;
    if (r >= numRows) {
      break;
    }
    float result = 0.0f;
    for (int fr = -filterWidth / 2; fr <= filterWidth / 2; fr++) {
      for (int fc = -filterWidth / 2; fc <= filterWidth / 2; fc++) {
	int ir = min(max(r + fr, 0), numRows - 1);
	int ic = min(max(c + fc, 0), numCols - 1);
	float ipx = inputChannel[ir * numCols + ic];
	float fpx = filter[(fr + filterWidth / 2) * filterWidth + fc + filterWidth / 2];
	result += ipx * fpx;
      }
    }
    outputChannel[r * numCols + c] = result;
  }
}

__global__ void gaussian_blur(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  constexpr int tileSize = 32;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= numCols) {
    return;
  }

  int rStart = blockIdx.y * tileSize;
  int rStop = min(rStart + tileSize, numRows);

  for (int r = rStart; r < rStop; r++) {
    float result = 0.0f;
    for (int fr = -filterWidth / 2; fr <= filterWidth / 2; fr++) {
      for (int fc = -filterWidth / 2; fc <= filterWidth / 2; fc++) {
	int ir = min(max(r + fr, 0), numRows - 1);
	int ic = min(max(c + fc, 0), numCols - 1);
	float ipx = inputChannel[ir * numCols + ic];
	float fpx = filter[(fr + filterWidth / 2) * filterWidth + fc + filterWidth / 2];
	result += ipx * fpx;
      }
    }
    outputChannel[r * numCols + c] = result;
  }
}

/**
 * 4096, 4096)  44.919 ms 60.5 gflops/s  0.7 gbytes/s
 */
__global__ void gaussian_blur_v1(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (c >= numCols || r >= numRows)
    return;
  
  float result = 0.0f;
  for (int fr = -filterWidth / 2; fr <= filterWidth / 2; fr++) {
    for (int fc = -filterWidth / 2; fc <= filterWidth / 2; fc++) {
      int ir = min(max(r + fr, 0), numRows - 1);
      int ic = min(max(c + fc, 0), numCols - 1);
      float ipx = inputChannel[ir * numCols + ic];
      float fpx = filter[(fr + filterWidth / 2) * filterWidth + fc + filterWidth / 2];
      result += ipx * fpx;
    }
  }
  outputChannel[r * numCols + c] = result;
}

/**
 * Put filter in shared memory.  This doesn't help.
 * (4096, 4096)  46.158 ms 58.9 gflops/s  0.7 gbytes/s
 */
__global__ void gaussian_blur_v2(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  extern __shared__ float s_filter[];
  
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  int fidx = threadIdx.x + blockDim.x * threadIdx.y;
  if (fidx < filterWidth * filterWidth) {
    s_filter[fidx] = filter[fidx];
  }
  __syncthreads();
  
  if (c >= numCols || r >= numRows)
    return;
  
  float result = 0.0f;
  for (int fr = -filterWidth / 2; fr <= filterWidth / 2; fr++) {
    for (int fc = -filterWidth / 2; fc <= filterWidth / 2; fc++) {
      int ir = min(max(r + fr, 0), numRows - 1);
      int ic = min(max(c + fc, 0), numCols - 1);
      float ipx = inputChannel[ir * numCols + ic];
      float fpx = s_filter[(fr + filterWidth / 2) * filterWidth + fc + filterWidth / 2];
      result += ipx * fpx;
    }
  }
  outputChannel[r * numCols + c] = result;
}

/**
 * Put image tile in shared memory.  Shit.
 * (4096, 4096)  54.074 ms 50.3 gflops/s  0.6 gbytes/s
 */
__global__ void gaussian_blur_v3(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  extern __shared__ float s_tile[];

  int tileCol = blockIdx.x * blockDim.x;
  int tileRow = blockIdx.y * blockDim.y;

  int tileWidth = blockDim.x + filterWidth - 1;
  int tileHeight = blockDim.y + filterWidth - 1;

  for (int threadRow = threadIdx.y; threadRow < tileHeight; threadRow += blockDim.y) {
    for (int threadCol = threadIdx.x; threadCol < tileWidth; threadCol += blockDim.x) {
      int imgRow = min(max(tileRow - filterWidth / 2 + threadRow, 0), numRows - 1);
      int imgCol = min(max(tileCol - filterWidth / 2 + threadCol, 0), numCols - 1);
      s_tile[threadRow * tileWidth + threadCol] = (float)inputChannel[imgRow * numCols + imgCol];
    }
  }
  __syncthreads();
  
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (c >= numCols || r >= numRows)
    return;
  
  float result = 0.0f;
  for (int fr = -filterWidth / 2; fr <= filterWidth / 2; fr++) {
    for (int fc = -filterWidth / 2; fc <= filterWidth / 2; fc++) {
      float filterPix = filter[(fr + filterWidth / 2) * filterWidth + fc + filterWidth / 2];
      float tilePix = s_tile[(threadIdx.y + fr + filterWidth / 2) * tileWidth + threadIdx.y + fc + filterWidth / 2];
      result += tilePix * filterPix;
    }
  }
  outputChannel[r * numCols + c] = result;
}

/*
 * Straight up tiled copy.  How fast is that?
 * (4096, 4096)  0.225 ms 12100.7 gflops/s  149.4 gbytes/s
 * Surprisingly slow (10% peak b/w) but way faster than my kernel.
 */
__global__ void gaussian_blur_tiled_copy(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (c >= numCols || r >= numRows)
    return;

  outputChannel[r * numCols + c] = inputChannel[r * numCols + c];
}

/*
 * Copying tile w/o halo to shared mem.
 * (4096, 4096)  0.330 ms 8225.3 gflops/s  101.5 gbytes/s
 * Still quite fast.
 */
__global__ void gaussian_blur_tiled_shmem(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (c >= numCols || r >= numRows)
    return;

  extern __shared__ float s_tile[];
  s_tile[threadIdx.y * blockDim.x + threadIdx.x] = (float)inputChannel[r * numCols + c];
  __syncthreads();

  outputChannel[r * numCols + c] = s_tile[threadIdx.y * blockDim.x + threadIdx.x];
}

/*
 * Actually do filterWidth^2 work.  Big slowdown.
 * (4096, 4096)  2.711 ms 1002.4 gflops/s  12.4 gbytes/s
 */
__global__ void gaussian_blur_fmas(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (c >= numCols || r >= numRows)
    return;

  /*
  extern __shared__ float s_tile[];
  s_tile[threadIdx.y * blockDim.x + threadIdx.x] = (float)inputChannel[r * numCols + c];
  __syncthreads();
  */
  float result = 0.0f;
  float f = 4.0; // s_tile[threadIdx.y * blockDim.x + threadIdx.x];
  float result0 = 0.0f;
  float result1 = 0.0f;
  float result2 = 0.0f;
  float result3 = 0.0f;
  float result4 = 0.0f;
  float result5 = 0.0f;
  float result6 = 0.0f;
  float result7 = 0.0f;
  float result8 = 0.0f;
    
  #pragma unroll 9
  for (int i = 0; i < 9; i++) {
    result0 += f * 2.0;
    result1 += f * 2.0;
    result2 += f * 2.0;
    result3 += f * 2.0;
    result4 += f * 2.0;
    result5 += f * 2.0;
    result6 += f * 2.0;
    result7 += f * 2.0;
    result8 += f * 2.0;
  }
  outputChannel[r * numCols + c] = result;
}

/*
 * Compute on the tile, but apply all fw^2 to the same pixel. And only
 * use filter[0].
 * (4096, 4096)  3.433 ms 791.7 gflops/s  9.8 gbytes/s
 */
__global__ void gaussian_blur_tile_fmas_only(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (c >= numCols || r >= numRows)
    return;

  extern __shared__ float s_tile[];
  s_tile[threadIdx.y * blockDim.x + threadIdx.x] = (float)inputChannel[r * numCols + c];
  __syncthreads();

  float result = 0.0f;
  float f = s_tile[threadIdx.y * blockDim.x + threadIdx.x];
    
  for (int i = 0; i < filterWidth; i++) {
    for (int j = 0; j < filterWidth; j++) {
      result += f * filter[0];
    }
  }
  outputChannel[r * numCols + c] = result;
}

/*
 * Same as above but actually use the whole filter.
 * (4096, 4096)  5.813 ms 467.6 gflops/s  5.8 gbytes/s
 */
__global__ void gaussian_blur_filter_fmas(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (c >= numCols || r >= numRows)
    return;

  extern __shared__ float s_tile[];
  s_tile[threadIdx.y * blockDim.x + threadIdx.x] = (float)inputChannel[r * numCols + c];
  __syncthreads();

  float result = 0.0f;
  float f = s_tile[threadIdx.y * blockDim.x + threadIdx.x];
    
  for (int i = 0; i < filterWidth; i++) {
    for (int j = 0; j < filterWidth; j++) {
      result += f * filter[filterWidth * i + j];
    }
  }
  outputChannel[r * numCols + c] = result;
}

/*
 * Same as prev but use a chunk of s_tile.
 * (4096, 4096)  8.968 ms 303.1 gflops/s  3.7 gbytes/s
 */
__global__ void gaussian_blur_filter_tile(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (c >= numCols || r >= numRows)
    return;

  extern __shared__ float s_tile[];
  s_tile[threadIdx.y * blockDim.x + threadIdx.x] = (float)inputChannel[r * numCols + c];
  __syncthreads();

  float result = 0.0f;
  float f = s_tile[threadIdx.y * blockDim.x + threadIdx.x];
    
  for (int i = 0; i < filterWidth; i++) {
    for (int j = 0; j < filterWidth; j++) {
      result += s_tile[filterWidth * i + j] * filter[filterWidth * i + j];
    }
  }
  outputChannel[r * numCols + c] = result;
}

/*
 */
__global__ void gaussian_blur_vN(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, int numRows,
                              int numCols, const float *const filter,
                              const int filterWidth) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  // extern __shared__ float s_filter[];
  // float* s_tile = s_filter + filterWidth * filterWidth;
  
  // if (threadIdx.x < filterWidth && threadIdx.y < filterWidth) {
  //   s_filter[threadIdx.y * filterWidth + threadIdx.x] = filter[threadIdx.y * filterWidth + threadIdx.x];
  // }

  extern __shared__ float s_tile[];
  if (c >= numCols || r >= numRows)
    return;

  s_tile[threadIdx.y * blockDim.x + threadIdx.x] = (float)inputChannel[r * numCols + c];
  __syncthreads();

  float result = 0.0f;
  float f = s_tile[threadIdx.y * blockDim.x + threadIdx.x];
    
  for (int i = 0; i < filterWidth; i++) {
    for (int j = 0; j < filterWidth; j++) {
      result += s_tile[filterWidth * i + j] * filter[filterWidth * i + j];
    }
  }
  outputChannel[r * numCols + c] = result;
}

// This kernel takes in an image represented as a uchar4 and splits
// it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4 *const inputImageRGBA,
                                 int numRows, int numCols,
                                 unsigned char *const redChannel,
                                 unsigned char *const greenChannel,
                                 unsigned char *const blueChannel) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < numCols && y < numRows) {
    int i = numCols * y + x;
    uchar4 rgba = inputImageRGBA[i];
    redChannel[i] = rgba.x;
    greenChannel[i] = rgba.y;
    blueChannel[i] = rgba.z;
  }
  // TODO
  //
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before
  // accessing GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
}

// This kernel takes in three color channels and recombines them
// into one image.  The alpha channel is set to 255 to represent
// that this image has no transparency.
__global__ void recombineChannels(const unsigned char *const redChannel,
                                  const unsigned char *const greenChannel,
                                  const unsigned char *const blueChannel,
                                  uchar4 *const outputImageRGBA, int numRows,
                                  int numCols) {
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  // make sure we don't try and access memory outside the image
  // by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue = blueChannel[thread_1D_pos];

  // Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;
extern float* h_filter__;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage,
                                const size_t numColsImage,
                                const float *const h_filter,
                                const size_t filterWidth) {

  // allocate memory for the three different channels
  // original
  checkCudaErrors(
      cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage *
                                           numColsImage));
  checkCudaErrors(
      cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

  // TODO:
  // Allocate memory for the filter on the GPU
  // Use the pointer d_filter that we have already declared for you
  // You need to allocate memory for the filter with cudaMalloc
  // be sure to use checkCudaErrors like the above examples to
  // be able to tell if anything goes wrong
  // IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  size_t filterSize = sizeof(float) * filterWidth * filterWidth;
  checkCudaErrors(cudaMalloc(&d_filter, filterSize));
		  
  // TODO:
  // Copy the filter on the host (h_filter) to the memory you just allocated
  // on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  // Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter__, filterSize, cudaMemcpyHostToDevice));
}

template<typename T, typename U>
T ceildiv(T p, U q) {
  return (p + q - 1) / q;
}

// Free all the memory that we allocated
// TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}

void your_gaussian_blur(const uchar4 *const h_inputImageRGBA,
                        uchar4 *const d_inputImageRGBA,
                        uchar4 *const d_outputImageRGBA, const size_t numRows,
                        const size_t numCols, unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred, const int filterWidth) {
  //allocateMemoryAndCopyToGPU(numRows, numCols, h_filter__, filterWidth);
  
  // TODO: Set reasonable block size (i.e., number of threads per block)
  int xBlock = 32;
  int yBlock = 32;
  const dim3 blockSize(xBlock, yBlock);

  // TODO:
  // Compute correct grid size (i.e., number of blocks per kernel launch)
  // from the image size and and block size.
  const dim3 gridSize(ceildiv(numCols, xBlock), ceildiv(numRows, yBlock));

  // TODO: Launch a kernel for separating the RGBA image into different color
  // channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
					    numRows,
					    numCols,
					    d_red,
					    d_green,
					    d_blue);
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // TODO: Call your convolution kernel here 3 times, once for each color
  // channel.
  auto constexpr ITERS = 5;
  GpuTimer t;
  auto bytes = numRows * numCols * sizeof(char) * 2;
  auto flops = numRows * numCols * filterWidth * filterWidth * 2;
  size_t filterBytes = filterWidth * filterWidth * sizeof(float);
  size_t tileBytes = (xBlock + filterWidth - 1) * (yBlock + filterWidth - 1) * sizeof(float);
  size_t shmemBytes = filterBytes + tileBytes;

  dim3 convGridSize(ceildiv(numCols, xBlock), ceildiv(numRows, yBlock));
  dim3 convBlockSize(xBlock);
  
  for (int i = 0; i < ITERS; i++) {
    t.Start();
    gaussian_blur<<<convGridSize, convBlockSize, shmemBytes>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
    t.Stop();
    auto ms = t.Elapsed();
    printf("(%lu, %lu)  %.3f ms %.1f gflops/s  %.1f gbytes/s\n", numRows, numCols, ms, flops/ms/1e6, bytes/ms/1e6);
  }
    
  gaussian_blur<<<convGridSize, convBlockSize, shmemBytes>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<convGridSize, convBlockSize, shmemBytes>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
  
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors()
  // immediately after launching your kernel to make sure that you didn't make
  // any mistakes.
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for
  // you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred,
                                             d_blueBlurred, d_outputImageRGBA,
                                             numRows, numCols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  cleanup();
}
