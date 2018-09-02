#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int v_k=50;
int v_n=250000;
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
int * getSerise(int n) {
	int *array = malloc(n * sizeof(int));
	for(int i=0;i<n;i++){
		*(array+i) = i+1;
	}
	return array;
}

// pick up random number from serise
int * getRandom(int k, int n){
	// create a serise from 1 ~ n
	int *serise = getSerise(n);
	// random select 
	int *rd = malloc(k * sizeof(int));
	int t=0;
	for(int i=0;i<k;i++){
		while(1){
			int flag = 0;
			// randomly number from range of 0 to n-1
			int rd_num = rand()%(n-1);
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
	return rd;
}
 
int main(){

	int seed = time(NULL);
    srand(seed);
    clock_t begin = clock();
	int *p = getRandom(v_k,v_n);
	printf("Output:");
	for(int i=0;i<v_k;i++){
		printf("%d ",p[i]);
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("\n Spend: %f s",time_spent);
}
