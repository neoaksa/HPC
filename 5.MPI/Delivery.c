// each of 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MASTER  0
#define TAG     0


// enlarge the array by size of "newsize"
void enlarge (int *arr, int newsize)
{
    arr = (int *)realloc(arr, sizeof(int)*newsize);
}

int main(int argc, char* argv[])
{

    int my_rank, num_nodes, *array;

    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    for(int iter=0; iter <2; iter++){      
        // start point
        if (my_rank == 0 && iter==0) {
            array = (int *)malloc(sizeof(int)*(1<<(1)));
            MPI_Send(array, 1<<(1), MPI_INT, my_rank+1, TAG, MPI_COMM_WORLD);
            printf("rank:%d(iter:%d) to %d, size of data:%d \n", my_rank,iter,my_rank+1, 1<<(1));
        }
        // receive, enlarge and send to next one
        array = (int *)malloc(sizeof(int)*(1<<(my_rank+iter*num_nodes)));
        int previous =  my_rank==0?(num_nodes-1):(my_rank-1);
        int next = my_rank==(num_nodes-1)?0:(my_rank+1);
        MPI_Recv(array, 1<<(my_rank+iter*num_nodes), MPI_INT, previous, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        enlarge(array, 1<<(my_rank+iter*num_nodes+1));
        MPI_Send(array, 1<<(my_rank+iter*num_nodes+1), MPI_INT, next, TAG, MPI_COMM_WORLD);
        printf("rank:%d(iter:%d) to %d, size of data:%d \n", my_rank,iter, next, 1<<(my_rank+iter*num_nodes+1));
    }

    MPI_Finalize();

    return 0;
}
