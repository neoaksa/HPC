

#include <stdio.h>
#include <cstdlib>
#include <sys/time.h>
using namespace std;
int main(int argc, char **argv)
{
	unsigned long  VECSIZE = (1 << 30);
	unsigned long long *vec;
   vec =(unsigned long long*) malloc(VECSIZE*sizeof(unsigned long long));
	 for (unsigned long i=0; i < VECSIZE; i++) {
      vec[i]=0;
	}
	 //timing 
	struct timeval start, finish;
	// timing start
	gettimeofday (&start, NULL);
	for(unsigned long  i=0;i<VECSIZE;i++){
		if(i< VECSIZE/2){
			vec[i] = (i+1)*(i%10+1);
		}
		else{
			vec[i] = (i+2*(VECSIZE/2-i))*(i%10+1);
		}
	}
	unsigned long long sum=0;
	for(unsigned long  i=0;i<VECSIZE;i++){
		sum+=vec[i];
	}
	
		//timing end
	gettimeofday (&finish, NULL);
	double elapsed = (finish.tv_sec - start.tv_sec)*1000000 + finish.tv_usec - start.tv_usec;
	
	printf("Time spent: %f \n",elapsed);
	printf("total: %llu",sum);
		
	return 0;
}

//~ 1585267071787204590
//~ 1585267071787204590 

