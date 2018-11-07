/* Diffusion Simulation 
 * nvcc -arch=sm_30 DS.cu -run
 * */
#include "common/book.h"
#include "common/cpu_bitmap.h"


#include "cuda.h"
#include "cuda_gl_interop.h"
#include <GL/freeglut_std.h>
#include <GL/freeglut_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>
// set a 3D volume
//define the data set size (cubic volume)
#define DATAXSIZE 256
#define DATAYSIZE 256
#define DATAZSIZE 256
//block size = 8*8*8 = 512
#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8
//time iteration
#define t 10000
//OpenGL version
#define DIM 256
//CPU Validation 
#define CPUV 0


PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

GLuint  bufferObj;
cudaGraphicsResource *resource;


// refresh shadow array
__global__ void refesh(float (*output_array)[DATAYSIZE][DATAXSIZE],
                          float (*shadow_array)[DATAYSIZE][DATAXSIZE],uchar4 *ptr)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idz = blockIdx.z*blockDim.z + threadIdx.z;
    // updae shadow and reset barrier
    if(idz==DATAZSIZE/2){
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int offset = x + y * blockDim.x * gridDim.x;
            if(output_array[idz][idy][idx]>10){
                ptr[offset].x = 0;
                ptr[offset].y = 0;
                ptr[offset].z = 100;
                ptr[offset].w = 255;}
            else if(output_array[idz][idy][idx]>1){
                ptr[offset].x = 100;
                ptr[offset].y = 0;
                ptr[offset].z = 100;
                ptr[offset].w = 255;
            }else if(output_array[idz][idy][idx]>0.1){
                ptr[offset].x = 100;
                ptr[offset].y = 100;
                ptr[offset].z = 0;
                ptr[offset].w = 255;
            }else if(output_array[idz][idy][idx]>0.01){
                ptr[offset].x = 50;
                ptr[offset].y = 50;
                ptr[offset].z = 50;
                ptr[offset].w = 255;
            }
        
    }
    shadow_array[idz][idy][idx] = output_array[idz][idy][idx];
}

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

static void draw_func( void ) {
    // we pass zero as the last parameter, because out bufferObj is now
    // the source, and the field switches from being a pointer to a
    // bitmap to now mean an offset into a bitmap object
    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );
    glDrawPixels( DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glutSwapBuffers();
}

static void key_func( unsigned char key, int x, int y ) {
    switch (key) {
        case 27:
            // clean up OpenGL and CUDA
            glutLeaveMainLoop();
    }
}

int main( int argc, char **argv ) 
{
     cudaDeviceProp  prop;
    int dev;

    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 0;
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );

    // tell CUDA which dev we will be using for graphic interop
    // from the programming guide:  Interoperability with OpenGL
    //     requires that the CUDA device be specified by
    //     cudaGLSetGLDevice() before any other runtime calls.

    HANDLE_ERROR( cudaGLSetGLDevice( dev ) );

    // these GLUT calls need to be made before the other OpenGL
    // calls, else we get a seg fault
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( DIM, DIM );
    glutCreateWindow( "bitmap" );

    glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

    // the first three are standard OpenGL, the 4th is the CUDA reg 
    // of the bitmap these calls exist starting in OpenGL 1.5
    glGenBuffers( 1, &bufferObj );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
    glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4,
                  NULL, GL_DYNAMIC_DRAW_ARB );

    HANDLE_ERROR( 
        cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone ) );


    
    
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
		fprintf(stderr,"malloc1 Fail \n"); 
	}
	// allocate storage for shadow arry
    if ((shadow_c = (nRarray *)malloc((nx*ny*nz)*sizeof(float))) == 0) {
		fprintf(stderr,"malloc1 Fail \n"); 
	}
    if ((output_cpu = (nRarray *)malloc((nx*ny*nz)*sizeof(float))) == 0) {
		fprintf(stderr,"malloc1 Fail \n"); 
	}
	// allocate storage for shadow arry
    if ((shadow_cpu = (nRarray *)malloc((nx*ny*nz)*sizeof(float))) == 0) {
		fprintf(stderr,"malloc1 Fail \n"); 
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
            // do work with the memory dst being on the GPU, gotten via mapping
        HANDLE_ERROR( cudaGraphicsMapResources( 1, &resource, NULL ) );
        uchar4* devPtr;
        size_t  size;
        HANDLE_ERROR( 
        cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, 
                                              &size, 
                                              resource) );
        diffusion<<<gridSize,blockSize>>>(output_d,shadow_d);
        refesh<<<gridSize,blockSize>>>(output_d,shadow_d,devPtr);
        HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &resource, NULL ) );
        printf("%d \n",time);
        // set up GLUT and kick off main loop
        glutKeyboardFunc( key_func );
        glutDisplayFunc( draw_func );
        glutPostRedisplay();
        glutMainLoopEvent();
    }

    // copy output data back to host
    result = cudaMemcpy(output_c, output_d, ((nx*ny*nz)*sizeof(float)), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
		fprintf(stderr, ("cudaMemcpy dev (block)->host failed."));
		exit(1);
	}
	

	//CPU validation is ON/OFF
	if(CPUV){
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
        myfile.open ("DS.csv",std::ios_base::app);
    //     myfile.open ("DS.csv");
        for (unsigned i=0; i<nz; i++)
        for (unsigned j=0; j<ny; j++)
            for (unsigned k=0; k<nx; k++){
                if( output_c[i][j][k]!=0)
                myfile << i << "," << j << "," << k << "," << output_c[i][j][k] << "," << std::to_string(t) << std::endl;
            }
        myfile.close();
    }
    

    
// 	// free memory
//     free(output_c);
//     free(shadow_c);
//     result = cudaFree(shadow_d);
//     if (result != cudaSuccess) {
// 		fprintf(stderr, ("Failed to Free device buffer - shadow_d"));
// 		exit(1);
// 	}
//     result = cudaFree(output_d);
//     if (result != cudaSuccess) {
// 		fprintf(stderr, ("Failed to Free device buffer - output_d"));
// 		exit(1);
// 	}


}


