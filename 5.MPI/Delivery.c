// each of 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define TAG     0
#define MAXITER 5
// #define DATA_SIZE 16

// enlarge the array by size of "newsize"
// void enlarge (unsigned long long *arr, int newsize)
// {
//     arr = (unsigned long long *)realloc(arr,   sizeof(unsigned long long)*newsize);
// }

int main(int argc, char* argv[])
{

    int my_rank, num_nodes;
    unsigned long long *array;
    int DATA_SIZE = atol(argv[1]);
    
    struct timeval start, finish;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    for(int iter=0; iter<MAXITER; iter++){      
        // start point
        if (my_rank == 0 && iter==0) {
            array = (unsigned long long *)malloc(sizeof(unsigned long long)*(1<<(DATA_SIZE)));
            // timing start
            gettimeofday (&start, NULL);
            MPI_Send(array, 1<<(DATA_SIZE), MPI_UNSIGNED_LONG_LONG, my_rank+1, TAG, MPI_COMM_WORLD);
            gettimeofday (&finish, NULL);
            double elapsed_cpu = (finish.tv_sec - start.tv_sec)*1000000 + finish.tv_usec - start.tv_usec; 
            printf("rank:%d(iter:%d) next: %d, previous:null, size of data:%d. time:%f\n", my_rank,iter,my_rank+1, 1<<(DATA_SIZE),elapsed_cpu);
        }
        else{
            // receive, enlarge and send to next one
            array = (unsigned long long *)malloc(sizeof(unsigned long long)*(1<<(DATA_SIZE)));
            int previous =  my_rank==0?(num_nodes-1):(my_rank-1);
            int next = my_rank==(num_nodes-1)?0:(my_rank+1);
            // the first node does't recv any thing in last iternative
            if(my_rank!=0 || iter!=(MAXITER-1)){
                MPI_Recv(array, 1<<(DATA_SIZE), MPI_UNSIGNED_LONG_LONG, previous, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // the last node of last iterative stops iternative
            if(iter != MAXITER-1 || my_rank != num_nodes -1){
//                 enlarge(array, 1<<(my_rank+iter*num_nodes+1));
                // timing start
                gettimeofday (&start, NULL);
                MPI_Send(array, 1<<(DATA_SIZE), MPI_UNSIGNED_LONG_LONG, next, TAG, MPI_COMM_WORLD);
                gettimeofday (&finish, NULL);
                double elapsed_cpu = (finish.tv_sec - start.tv_sec)*1000000 + finish.tv_usec - start.tv_usec; 
                printf("rank:%d(iter:%d) next: %d previous:%d, size of data:%d. time:%f \n", my_rank,iter, next,previous, 1<<(DATA_SIZE),elapsed_cpu);
            }
        }
    }

    MPI_Finalize();

    return 0;
}
