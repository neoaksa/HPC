/*
 * workperform.cu
 * generate vector 
 * execute dot product for two vectors
 * reference: "Optimizing Parallel Reduction in CUDA" by Mark Harris
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

template <unsigned int block_Size>
__device__ void warpReduce(volatile unsigned long long *vec, unsigned int tid) {
	if (block_Size >= 64) vec[tid] += vec[tid + 32];
	if (block_Size >= 32) vec[tid] += vec[tid + 16];
	if (block_Size >= 16) vec[tid] += vec[tid + 8];
	if (block_Size >= 8) vec[tid] += vec[tid + 4];
	if (block_Size >= 4) vec[tid] += vec[tid + 2];
	if (block_Size >= 2) vec[tid] += vec[tid + 1];
}

// The __global__ directive identifies this function as a kernel
// Note: all kernels must be declared with return type void 
template <unsigned int block_Size>
__global__ void cu_vector_dot (unsigned long long *output_d, unsigned long  vecSize)
{
	// genrate vector for F dot produt D
	__shared__ unsigned long long vec[block_Size];
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned long  gid = bid*block_Size+tid;
    vec[tid] = 0;
    if(gid< vecSize/2){
		vec[tid] = (gid+1)*(gid%10+1);
	}
	else{
		vec[tid] = (gid+2*(vecSize/2-gid))*(gid%10+1);
	}
	 __syncthreads();
	//unroll the iterative
	if (block_Size >= 512) {
		if (tid < 256) { vec[tid] += vec[tid + 256]; } __syncthreads(); }
	if (block_Size >= 256) {
		if (tid < 128) { vec[tid] += vec[tid + 128]; } __syncthreads(); }
	if (block_Size >= 128) {
		if (tid < 64) { vec[tid] += vec[tid + 64]; } __syncthreads(); }
	// when tid<=32, it's in a warp, we don't need syncthreads, it's sequence
	if (tid < 32) {		
		warpReduce<block_Size>(vec, tid);
	} 
	if (tid == 0) output_d[bid] = vec[0];
}


// This function is called from the host computer.
// It manages memory and calls the function that is executed on the GPU
 void innerproduct(unsigned long long *output,  unsigned long vecSize, int blockSize)
{
	// block_d and thread_d are the GPU counterparts of the arrays that exists in host memory 
	unsigned long long *output_d;
	int blockNum = vecSize/blockSize;
	cudaError_t result;

	// allocate space in the device 
	result = cudaMalloc ((void**) &output_d, sizeof(unsigned long long) * blockNum);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (block) failed.");
		exit(1);
	}

	
	//copy the arrays from host to the device 
	result = cudaMemcpy (output_d, output, sizeof(unsigned long long) * blockNum, cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host->dev (block) failed.");
		exit(1);
	}

	
	// set execution configuration
	dim3 dimBlock (blockSize);
	dim3 dimGrid (blockNum);
	
	// actual computation: Call the kernel
	switch (blockSize)
	{
		case 512:
		cu_vector_dot<512><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
		case 256:
		cu_vector_dot<256><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
		case 128:
		cu_vector_dot<128><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
		case 64:
		cu_vector_dot< 64><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
		case 32:
		cu_vector_dot< 32><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
		case 16:
		cu_vector_dot< 16><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
		case 8:
		cu_vector_dot< 8><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
		case 4:
		cu_vector_dot< 4><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
		case 2:
		cu_vector_dot< 2><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
		case 1:
		cu_vector_dot< 1><<< dimGrid, dimBlock >>>(output_d, vecSize); break;
	}

	// transfer results back to host
	result = cudaMemcpy (output, output_d, sizeof(unsigned long long) * blockNum, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host <- dev (block) failed.");
		exit(1);
	}

	// release the memory on the GPU 
	result = cudaFree (output_d);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaFree (block) failed.");
		exit(1);
	}
}

