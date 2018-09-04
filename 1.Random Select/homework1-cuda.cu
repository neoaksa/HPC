#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <time.h>

#define DATA_SIZE 25000000
#define KSELECT 50
#define BLOCK_NUM 1
#define THREAD_NUM 50

// Check CUDA
bool InitCUDA()
{
	int count;
	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
			break;
			}
		}
	}
	if(i == count) {
	fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
	return false;
	}
	cudaSetDevice(i);
	return true;
}

__global__ static void randomSelect(int* result){
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	int size = (int)(DATA_SIZE/KSELECT);
	// generate number
	int start = bid*THREAD_NUM+tid*size+1;
	// random select
	curandState state;
	curand_init(tid,tid,0,&state);
	result[tid] = curand(&state)%(size)+start;
}

int main()
{
	if(!InitCUDA()) {
		return 0;
}
	printf("CUDA initialized.\n");

	int*result;
	clock_t begin = clock();
	cudaMalloc((void**) &result, sizeof(int) * KSELECT);
	randomSelect<<<BLOCK_NUM, THREAD_NUM,0>>>(result);
	int result_host[BLOCK_NUM];
	cudaMemcpy(&result_host, result, sizeof(int) * KSELECT,cudaMemcpyDeviceToHost);
	cudaFree(result);
	clock_t end = clock();
	// count time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("\n Spend: %f s",time_spent);
	for(int i = 0; i < KSELECT; i++) {
		printf("%d - %d\n",i,result_host[i]);
	}

	return 0;
}
 
