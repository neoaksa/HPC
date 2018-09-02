#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// create a serise from 1 ~ n
int v_k=50;
int v_n=250000;
//~ int *serise;

// struct type for transfering arguments to function
typedef struct thread_data {
   int k;
   int start;
   int end;
   int *result;
 
} thread_data;


// compare for sort
int comp (const void * elem1, const void * elem2) 
{
    int f = *((int*)elem1);
    int s = *((int*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

// generate numberic serise
int * getSerise(int start,int end) {
	int *array = malloc((end-start+1) * sizeof(int));
	for(int i=0;i<=(end-start);i++){
		*(array+i) = start+i;
	}
	return array;
}

// pick up random number from serise
void * getRandom(void *arg){
	thread_data *tdata=(thread_data *)arg;

    int k=tdata->k;
    int start=tdata->start;
    int end=tdata->end;
	// create serise
	int *serise = getSerise(start,end);
	// random select 
	int *rd = malloc(k * sizeof(int));
	int t=0;

	for(int i=0;i<k;i++){
		while(1){		
			int flag = 0;
			// randomly number from range of 0 to n-1
			int rd_num = (rand()%(end-start));
			// check duplicate
			for(int j=0;j<t;j++){
				if(rd[j]==serise[rd_num]){
					flag = 1;
					break;
				}
			}
			if(flag==0){
				rd[i] = serise[rd_num];
				t++;
				break;
			}
		}
	}
	// sort array
	qsort(rd, k, sizeof(int), comp);
	tdata->result = rd;
	return 0;
	//~ pthread_exit(NULL);
}
 
int main(){
	int seed = time(NULL);
    srand(seed);
	clock_t begin = clock();
    //~ serise = getSerise(v_n);
	// setting threads
	int thread_num = 5; 
	pthread_t tid[thread_num];
	thread_data tdata[thread_num];
	int silice_size = (int)(v_n/thread_num);
	// create threads
	for(int i=0; i<thread_num;i++){
		tdata[i].k=(int)(v_k/thread_num);
		tdata[i].start=i*(silice_size);
		tdata[i].end=(i+1)*silice_size-1;
		tdata[i].result=malloc((int)(v_k/thread_num)*sizeof(int));
		pthread_create(&tid[i], NULL, getRandom, (void *)&tdata[i]);
	}
	// join threads
	for(int i=0;i<thread_num;i++){
		pthread_join(tid[i], NULL);
	}
	// output
	clock_t end = clock();
	int *p = malloc(v_k*sizeof(int));
	int t=0;
	printf("Output:");
	for(int n=0; n<thread_num;n++){
		for(int i=0;i<(int)(v_k/thread_num);i++){
			*(p+t)=tdata[n].result[i];
			t++;
			printf("%d ",tdata[n].result[i]);
		}	
	}
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("\n cost %f s",time_spent);
	return 0;
}
