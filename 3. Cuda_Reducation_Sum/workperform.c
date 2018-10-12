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
#define BLOCK_SIZE 64
//~ #define VECSIZE 1 << 10

// The function innerproduct is in the file blockAndThread.cu
extern void innerproduct(int *output,  int vecSize, int blockSize);

int main (int argc, char *argv[])
{	
   int VECSIZE = (1 << 6);
   // Declare arrays and initialize to 0
   int blockNum = VECSIZE/BLOCK_SIZE;
   int output[blockNum];
   int i;
   for (i=0; i < blockNum; i++) {
      output[i]=0;
   }

   
   // Call the function that will call the GPU function
   innerproduct (output, VECSIZE, BLOCK_SIZE);
 
   // Again, print the arrays
   printf ("Final state of the array blockId:\n");
   unsigned long  long sum = 0;
   for (i=0; i < blockNum; i++) {
      printf ("%d ", output[i]);
      sum += output[i];
   }
   printf ("%llu ", sum);
   printf ("\n");
   return 0;
}
