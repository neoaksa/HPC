/* Diffusion Simulation 
 * nvcc -arch=sm_30 DS.cu -run
 * */

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>
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
#define t 10000


// device function to set the 3D volume
__global__ void diffusion(float (*output_array)[DATAYSIZE][DATAXSIZE],
                          float (*shadow_array)[DATAYSIZE][DATAXSIZE])
{   
//     // get grid, only works on GTX 1000 up
//     grid_group g = this_grid();
    // get position
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idz = blockIdx.z*blockDim.z + threadIdx.z;
    
    // not the edge
    if(idx>0 && idx<DATAXSIZE-1 && idy>0 && idy<DATAYSIZE-1 && idz>0 && idz<DATAZSIZE-1){
        output_array[idz][idy][idx] = (shadow_array[idz][idy][idx-1] + shadow_array[idz][idy][idx+1]
                                    + shadow_array[idz][idy-1][idx] + shadow_array[idz][idy+1][idx]
                                    + shadow_array[idz-1][idy][idx] + shadow_array[idz+1][idy][idx])/6;
    }
    // reach to the edge to rebound
    else{
        int nbr = 6;
        if(idx==0 || idx==DATAXSIZE-1) nbr-=1;
        if(idy==0 || idy==DATAYSIZE-1) nbr-=1;
        if(idz==0 || idz==DATAZSIZE-1) nbr-=1;
        output_array[idz][idy][idx] = (((idx==0)? 0:shadow_array[idz][idy][idx-1]) + ((idx==(DATAXSIZE-1))? 0: shadow_array[idz][idy][idx+1])
                                    + ((idy==0)? 0:shadow_array[idz][idy-1][idx]) + ((idy==(DATAYSIZE-1))? 0: shadow_array[idz][idy+1][idx])
                                    + ((idz==0)? 0:shadow_array[idz-1][idy][idx])+ ((idz==(DATAZSIZE-1))? 0: shadow_array[idz+1][idy][idx]))/nbr;
    }
}

// refresh shadow array
__global__ void refesh(float (*output_array)[DATAYSIZE][DATAXSIZE],
                          float (*shadow_array)[DATAYSIZE][DATAXSIZE])
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idz = blockIdx.z*blockDim.z + threadIdx.z;
    // updae shadow and reset barrier
    shadow_array[idz][idy][idx] = output_array[idz][idy][idx];
}

// cpu version for validation
void diffusion_cpu(float (*output_array)[DATAYSIZE][DATAXSIZE],
                   float (*shadow_array)[DATAYSIZE][DATAXSIZE])
{   
    for(int time=1; time<=t; time++){
        for(int idz=0;idz<DATAZSIZE;idz++)
            for(int idy=0;idy<DATAYSIZE;idy++)
                for(int idx=0;idx<DATAXSIZE;idx++){
                    if(idx>0 && idx<DATAXSIZE-1 && idy>0 && idy<DATAYSIZE-1 && idz>0 && idz<DATAZSIZE-1){
                        output_array[idz][idy][idx] = (shadow_array[idz][idy][idx-1] + shadow_array[idz][idy][idx+1]
                                                    + shadow_array[idz][idy-1][idx] + shadow_array[idz][idy+1][idx]
                                                    + shadow_array[idz-1][idy][idx] + shadow_array[idz+1][idy][idx])/6;
                    }
                    // reach to the edge to rebound
                    else{
                        int nbr = 6;
                        if(idx==0 || idx==DATAXSIZE-1) nbr-=1;
                        if(idy==0 || idy==DATAYSIZE-1) nbr-=1;
                        if(idz==0 || idz==DATAZSIZE-1) nbr-=1;
                        output_array[idz][idy][idx] = (((idx==0)? 0:shadow_array[idz][idy][idx-1]) + ((idx==(DATAXSIZE-1))? 0: shadow_array[idz][idy][idx+1])
                                                    + ((idy==0)? 0:shadow_array[idz][idy-1][idx]) + ((idy==(DATAYSIZE-1))? 0: shadow_array[idz][idy+1][idx])
                                                    + ((idz==0)? 0:shadow_array[idz-1][idy][idx])+ ((idz==(DATAZSIZE-1))? 0: shadow_array[idz+1][idy][idx]))/nbr;
//                         printf("%d,%d,%d-%f \n", idz,idy,idx, output_array[idz][idy][idx]);
                    }
                }
        // updae shadow and reset barrier/signal
        for(int idz=0;idz<DATAZSIZE;idz++)
            for(int idy=0;idy<DATAYSIZE;idy++)
                for(int idx=0;idx<DATAXSIZE;idx++){
                    shadow_array[idz][idy][idx] = output_array[idz][idy][idx];
                }
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
    unsigned int init_pos[6] = {0};
    init_pos[0] 	= DATAZSIZE/2 - 2;// x min
    init_pos[1] 	= DATAZSIZE/2 + 2;// x max
    init_pos[2] 	= DATAYSIZE/2 - 2;// y min
    init_pos[3] 	= DATAYSIZE/2 + 2;// y max
    init_pos[4] 	= DATAXSIZE/2 - 2;// z min
    init_pos[5] 	= DATAXSIZE/2 + 2;// z max
    //initial concetration
    const float con_begin = 20000.0;
    
    // pointers for data set storage via malloc
    nRarray *output_c; // storage for result stored on host
    nRarray *output_d; // storage for result computed on device
    nRarray *shadow_c; // shadow array for saving temp value on host
    nRarray *shadow_d; // shadow array for saving temp value
    nRarray *output_cpu; // for cpu version
    nRarray *shadow_cpu; // for cpu version
    // allocate storage for receiving output
    if ((output_c = (nRarray *)malloc((nx*ny*nz)*sizeof(float))) == 0) {
		fprintf(stderr,"malloc1 Fail \n"); return 1;
	}
	// allocate storage for shadow arry
    if ((shadow_c = (nRarray *)malloc((nx*ny*nz)*sizeof(float))) == 0) {
		fprintf(stderr,"malloc1 Fail \n"); return 1;
	}
    if ((output_cpu = (nRarray *)malloc((nx*ny*nz)*sizeof(float))) == 0) {
		fprintf(stderr,"malloc1 Fail \n"); return 1;
	}
	// allocate storage for shadow arry
    if ((shadow_cpu = (nRarray *)malloc((nx*ny*nz)*sizeof(float))) == 0) {
		fprintf(stderr,"malloc1 Fail \n"); return 1;
	}
    // inital concetration
    for(int k=init_pos[0];k<=init_pos[1];k++)
        for(int j=init_pos[2]; j<=init_pos[3];j++) 
            for(int i=init_pos[4];i<=init_pos[5];i++ ){
                output_c[k][j][i] = con_begin;
                shadow_c[k][j][i] = con_begin;
                output_cpu[k][j][i] = con_begin;
                shadow_cpu[k][j][i] = con_begin;
    }
    // allocate GPU device buffers
    result = cudaMalloc((void **) &output_d, (nx*ny*nz)*sizeof(float));
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to allocate device buffer-output_d"));
		exit(1);
	}
    result = cudaMalloc((void **) &shadow_d, (nx*ny*nz)*sizeof(float));
    if (result != cudaSuccess) {
		fprintf(stderr, ("Failed to allocate device buffer- shadow_d"));
		exit(1);
	}
	// copy host to device
	result = cudaMemcpy(shadow_d, shadow_c, (nx*ny*nz)*sizeof(float), cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host->dev (block) failed.");
		exit(1);
	}
    result = cudaMemcpy(output_d, output_c, (nx*ny*nz)*sizeof(float), cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host->dev (block) failed.");
		exit(1);
	}
	
    //timing 
	struct timeval start_cpu, finish_cpu,start_gpu, finish_gpu;
	// timing start
	gettimeofday (&start_gpu, NULL);
    
    // compute result
    const dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 gridSize((DATAXSIZE/BLKXSIZE), (DATAYSIZE/BLKYSIZE), (DATAZSIZE/BLKZSIZE));
    // loop with time t
    for(int time=1; time<=t; time++){
        diffusion<<<gridSize,blockSize>>>(output_d,shadow_d);
        refesh<<<gridSize,blockSize>>>(output_d,shadow_d);
//     	cudaDeviceSynchronize();
    }
    // copy output data back to host
    result = cudaMemcpy(output_c, output_d, ((nx*ny*nz)*sizeof(float)), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
		fprintf(stderr, ("cudaMemcpy dev (block)->host failed."));
		exit(1);
	}
    //timing end
	gettimeofday (&finish_gpu, NULL);
	double elapsed_gpu = (finish_gpu.tv_sec - start_gpu.tv_sec)*1000000 + finish_gpu.tv_usec - start_gpu.tv_usec;
    
    // timing start
	gettimeofday (&start_cpu, NULL);
    //cpu version
    diffusion_cpu(output_cpu,shadow_cpu);
    //timing end
	gettimeofday (&finish_cpu, NULL);
	double elapsed_cpu = (finish_cpu.tv_sec - start_cpu.tv_sec)*1000000 + finish_cpu.tv_usec - start_cpu.tv_usec;   
    
    //check two version
    for (unsigned i=0; i<nz; i++)
      for (unsigned j=0; j<ny; j++)
        for (unsigned k=0; k<nx; k++){
            if(output_c[i][j][k]!=output_cpu[i][j][k])
            {
                printf("check error happen \n");
                printf("position:[%d][%d][%d]. value:%f:%f \n",i,j,k ,output_c[i][j][k],output_cpu[i][j][k]);
//                 return 0;
                
            }
        }
    
    	
	printf("Time spent(GPU) of dt = %d: %f \n",t,elapsed_gpu);	
	printf("Time spent(CPU) of dt = %d: %f \n",t,elapsed_cpu);

    //write result to file
	std::ofstream myfile;
	myfile.open ("DS-2.csv",std::ios_base::app);
//     myfile.open ("DS.csv");
    for (unsigned i=0; i<nz; i++)
      for (unsigned j=0; j<ny; j++)
        for (unsigned k=0; k<nx; k++){
            if( output_c[i][j][k]!=0)
            myfile << i << "," << j << "," << k << "," << output_c[i][j][k] << "," << std::to_string(t) << std::endl;
        }
	myfile.close();
	// free memory
    free(output_c);
    free(shadow_c);
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

    return 0;
}


