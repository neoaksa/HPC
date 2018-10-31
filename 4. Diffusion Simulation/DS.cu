/* Diffusion Simulation 
 * nvcc -arch=sm_30 DS.cu -run
 * To compile with: nvcc -O2 -o DS DS.cu
 * reference:Inter-Block GPU Communication via Fast Barrier Synchronization
 * nvcc -arch compute_30 DS.cu -o DS
 * */

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>


// set a 3D volume
//define the data set size (cubic volume)
#define DATAXSIZE 64
#define DATAYSIZE 64
#define DATAZSIZE 64
//block size = 8*8*8 = 512
#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8
//time iteration
#define t 10


// device function to set the 3D volume
__global__ void diffusion(unsigned int *init_pos, float (*output_array)[DATAYSIZE][DATAXSIZE],
                          float (*shadow_array)[DATAYSIZE][DATAXSIZE], float con_begin, int *barrier, int signal)
{   
//     // get grid, only works on GTX 1000 up
//     grid_group g = this_grid();
    // get position
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
    // unique block id and tid
    unsigned bid = blockIdx.x 
                + blockIdx.y*gridDim.x 
                + blockIdx.z*gridDim.x*gridDim.y;
//     unsigned tid = idx 
//                 + (idy * gridDim.x)
//                 + (idz * gridDim.x * gridDim.y);
    unsigned tid_in_block = threadIdx.x
                            + threadIdx.y * blockDim.x
                            + threadIdx.z * blockDim.y * blockDim.x;
    unsigned blockNum = gridDim.x*gridDim.y*gridDim.z;
    // inital concetration
    if(idx>=init_pos[0] && idx<=init_pos[1] 
        && idy>=init_pos[2] && idy<=init_pos[3] 
        && idz>=init_pos[4] && idz<=init_pos[5] ){
            output_array[idz][idy][idx] = con_begin;
            shadow_array[idz][idy][idx] = con_begin;
        }
     __syncthreads();
    // diffusion in 3D space 
    // U(t+1) equal to avg of U(t) of six neigbors
    for(int i=1;i<=t;i++){
        //refresh array
        if(idx>0 && idx<DATAXSIZE-1 && idy>0 && idy<DATAYSIZE-1 && idz>0 && idz<DATAZSIZE-1){
            output_array[idz][idy][idx] = (shadow_array[idz][idy][idx-1] + shadow_array[idz][idy][idx+1]
                                        + shadow_array[idz][idy-1][idx] + shadow_array[idz][idy+1][idx]
                                        + shadow_array[idz-1][idy][idx] + shadow_array[idz+1][idy][idx])/6;
        }
//         //sync within grid
//         g.sync();
         __syncthreads();
        //sync between blocks
        // the first thread in each block takes charge of update barrier
        if ( tid_in_block == 0 )
            barrier[bid]=1;
        
        // the first N thread in block 1 response to check barrier? Problem here when block num > block size
        if(bid==1){
           if(tid_in_block < blockNum){
                while(barrier[tid_in_block]!=1);//{printf("%d-%d \n",bid,tid_in_block);};
            }
            __syncthreads();
            // ready to refresh array
            if(tid_in_block == 0)
                signal=1; 
        }
        //all other threads wait for the signal
        if(tid_in_block==0){
            while(signal!=1);//{printf("%d-%d \n",tid_in_block,bid);};
        }
        __syncthreads();
        // updae shadow and reset barrier/signal
        shadow_array[idz][idy][idx] = output_array[idz][idy][idx];
        if(bid==1){
            if(tid_in_block < blockNum){
                barrier[tid_in_block]=0;
            }
            __syncthreads();
            if(tid_in_block == 0)
                signal=0; 
        }
        if(tid_in_block==0){    
            while(signal!=0);
        }
        __syncthreads();
//         
    }
}

int main(int argc, char *argv[])
{
    typedef float nRarray[DATAYSIZE][DATAXSIZE];
    // overall data set sizes
    const int nx = DATAXSIZE;
    const int ny = DATAYSIZE;
    const int nz = DATAZSIZE;
    // error code
    cudaError_t result;
    // initial position
    unsigned int init_position[6] = {0};
    init_position[0] 	= DATAXSIZE/2 - 2;// x min
    init_position[1] 	= DATAXSIZE/2 + 2;// x max
    init_position[2] 	= DATAYSIZE/2 - 2;// y min
    init_position[3] 	= DATAYSIZE/2 + 2;// y max
    init_position[4] 	= DATAZSIZE/2 - 2;// z min
    init_position[5] 	= DATAZSIZE/2 + 2;// z max
    //initial concetration
    const float con_begin = 200.0;
    // pointers for data set storage via malloc
    unsigned int *inital_d; // initial the start status in device
    nRarray *output_c; // storage for result stored on host
    nRarray *output_d;  // storage for result computed on device
    nRarray *shadow_d; // shadow array for saving temp value
    int *barrier; // array for sync blocks
    int signal=0; // array for sync blocks(stop)
    // allocate storage for receiving output
    if ((output_c = (nRarray *)malloc((nx*ny*nz)*sizeof(float))) == 0) {
		fprintf(stderr,"malloc1 Fail \n"); return 1;
	}
    // allocate GPU device buffers
	result = cudaMalloc((void **) &inital_d, (6)*sizeof(unsigned int));
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to allocate device buffer-inital_d"));
		exit(1);
	}
    result = cudaMalloc((void **) &output_d, (int)(nx*ny*nz)*sizeof(float));
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to allocate device buffer-output_d"));
		exit(1);
	}
    result = cudaMalloc((void **) &shadow_d, (int)(nx*ny*nz)*sizeof(float));
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to allocate device buffer- shadow_d"));
		exit(1);
	}
	result = cudaMalloc((void **) &barrier, ((int)(DATAXSIZE*DATAYSIZE*DATAZSIZE)/(int)(BLKXSIZE*BLKYSIZE*BLKZSIZE))*sizeof(int));
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to allocate device buffer-barrier"));
		exit(1);
	}
	// copy host to device
	result = cudaMemcpy(inital_d, init_position, ((6)*sizeof(float)), cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host->dev (block) failed.");
		exit(1);
	}
    // compute result
    const dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 gridSize((DATAXSIZE/BLKXSIZE), (DATAYSIZE/BLKYSIZE), (DATAZSIZE/BLKZSIZE));
    
    diffusion<<<gridSize,blockSize>>>(inital_d,output_d,shadow_d,con_begin,barrier,signal);
    // copy output data back to host
    result = cudaMemcpy(output_c, output_d, ((int)(nx*ny*nz)*sizeof(float)), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
		fprintf(stderr, ("cudaMemcpy dev (block)->host failed."));
		exit(1);
	}
	cudaDeviceSynchronize();
//     cudaDeviceReset();
// 	//TEST
    for (unsigned i=0; i<nz; i++)
      for (unsigned j=0; j<ny; j++)
        for (unsigned k=0; k<nx; k++){
            if(output_c[i][j][k]!=0)
            printf("%d-%f ",i+j+k,output_c[i][j][k]);
        }
	
	// free memory
    free(output_c);
    result = cudaFree(shadow_d);
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to Free device buffer - shadow_d"));
		exit(1);
	}
    result = cudaFree(output_d);
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to Free device buffer - output_d"));
		exit(1);
	}
	result = cudaFree(inital_d);
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to Free device buffer - inital_d"));
		exit(1);
	}
		result = cudaFree(barrier);
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to Free device buffer - barrier"));
		exit(1);
	}
    return 0;
}
