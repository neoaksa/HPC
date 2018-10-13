/*
 * workperform.c
 * A "driver" program that calls a routine (i.e. a kernel)
 * that executes on the GPU.  The kernel fills two int arrays
 * with the block ID and the thread ID
 *
 * Note: the kernel code is found in the file 'workperform.cu'
 * compile both driver code and kernel code with nvcc, as in:
 * 			nvcc workperform.c workperform.cu
 */

#include <stdio.h>
#include <sys/time.h>
#include <cstdlib>

#define BLOCK_SIZE 512
//~ #define VECSIZE 1 << 10

// The function innerproduct is in the file blockAndThread.cu
extern void innerproduct(unsigned long long *output,  unsigned long  vecSize, int blockSize);

int main (int argc, char *argv[])
{	
   unsigned long VECSIZE = (1 << 30);
   // Declare arrays and initialize to 0
   int blockNum = VECSIZE/BLOCK_SIZE;
   unsigned long long *output;
   output =(unsigned long long*) malloc(blockNum*sizeof(unsigned long long));
   int i;
   for (i=0; i < blockNum; i++) {
      output[i]=0;
   }

   //timing 
	struct timeval start, finish;
	// timing start
	gettimeofday (&start, NULL);
	
   // Call the function that will call the GPU function
   innerproduct (output, VECSIZE, BLOCK_SIZE);
 
	//timing end
	gettimeofday (&finish, NULL);
	double elapsed = (finish.tv_sec - start.tv_sec)*1000000 + finish.tv_usec - start.tv_usec;
	
	printf("Time spent: %f \n",elapsed);
 
   // Again, print the arrays
   printf ("Final state of the array blockId:\n");
   unsigned long  long sum = 0;
   for (i=0; i < blockNum; i++) {
      //~ printf ("%d ", output[i]);
      sum += output[i];
   }
   printf ("%llu ", sum);
   printf ("\n");
   return 0;
}
