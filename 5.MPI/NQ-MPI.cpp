// Version 3.2 (2-core)
// <------  N-Queens Solutions  -----> <---- time ---->
//  N:           Total          Unique days hh:mm:ss.--
//  5:              10               2             0.00
//  6:               4               1             0.00
//  7:              40               6             0.00
//  8:              92              12             0.00
//  9:             352              46             0.00
// 10:             724              92             0.00
// 11:            2680             341             0.00
// 12:           14200            1787             0.00
// 13:           73712            9233             0.02
// 14:          365596           45752             0.05
// 15:         2279184          285053             0.22
// 16:        14772512         1846955             1.47
// 17:        95815104        11977939             9.42
// 18:       666090624        83263591          1:11.21
// 19:      4968057848       621012754          8:32.54
// 20:     39029188884      4878666808       1:10:55.48
// 21:    314666222712     39333324973       9:24:40.50



/* C/C++ program to solve N Queen Problem using 
backtracking */
#include<bits/stdc++.h> 
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 9
#define MASTER  0
#define TAG     0

// solution number
int TOTAL;  

  
/* A utility function to check if a queen can 
be placed on board[row][col]. Note that this 
function is called when "col" queens are 
already placed in columns from 0 to col -1. 
So we need to check only left side for 
attacking queens */
bool isSafe(int board[N][N], int row, int col) 
{ 
    int i, j; 
  
    /* Check this row on left side */
    for (i = 0; i < col; i++) 
        if (board[row][i]) 
            return false; 
  
    /* Check upper diagonal on left side */
    for (i=row, j=col; i>=0 && j>=0; i--, j--) 
        if (board[i][j]) 
            return false; 
  
    /* Check lower diagonal on left side */
    for (i=row, j=col; j>=0 && i<N; i++, j--) 
        if (board[i][j]) 
            return false; 
  
    return true; 
} 
  
/* A recursive utility function to solve N 
Queen problem */
bool solveNQUtil(int board[N][N], int col) 
{ 
    /* base case: If all queens are placed 
    then return true */
    if (col == N) 
    { 
        ++TOTAL;
        return true;
    } 
  
    /* Consider this column and try placing 
    this queen in all rows one by one */
    bool res = false; 
    for (int i = 0; i < N; i++) 
    { 
        /* Check if queen can be placed on 
        board[i][col] */
        if ( isSafe(board, i, col) ) 
        { 
            /* Place this queen in board[i][col] */
            board[i][col] = 1; 
  
            // Make result true if any placement 
            // is possible 
            res = solveNQUtil(board, col + 1) || res; 
  
            /* If placing queen in board[i][col] 
            doesn't lead to a solution, then 
            remove queen from board[i][col] */
            board[i][col] = 0; // BACKTRACK 
        } 
    } 
  
    /* If queen can not be place in any row in 
        this column col then return false */
    return res; 
} 
  
/* This function solves the N Queen problem using 
Backtracking. It mainly uses solveNQUtil() to 
solve the problem. It returns false if queens 
cannot be placed, otherwise return true and 
prints placement of queens in the form of 1s. 
Please note that there may be more than one 
solutions, this function prints one of the 
feasible solutions.*/
int solveNQ(int startRow) 
{ 
    int board[N][N]; 
    TOTAL = 0;
    memset(board, 0, sizeof(board)); 
    
    //set startRow strict
    board[startRow][0] = 1;
  
    if (solveNQUtil(board, 1) == false) 
    { 
        printf("Solution does not exist"); 
    } 
  
    return TOTAL; 
} 

// // check pool empty
// bool checkPoolEmpty(int *pool){
//     for(int i=0; i<sizeof(pool)/sizeof(*pool); i++){
//             if(pool[i]==0)
//                 return false;
//     }
//     return true;
// }
  
// driver program to test above function 
int main(int argc, char* argv[]) 
{ 
    int my_rank, num_nodes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    

    //buff save solution
    int node_solution[num_nodes-1] = {0};
    MPI_Request rq[num_nodes-1];
//     MPI_Request rq_rec[num_nodes-1];
    // Master node send the task to idle salve nodes
    if(my_rank==MASTER){
        int total_solution = 0;
        int row = 0;
        int flag[num_nodes-1] = {0}; //async flag
        int wait[num_nodes-1] = {0}; //Irecv wait flag
        //node pool
        int pool[num_nodes-1] = {0};
        //iternate all rows in first column
        while(row<N){
            //find idle node, send out task
            for(int node=0; node < (num_nodes-1)&&row<N; node++){
                if(pool[node]==0){
                    MPI_Send(&row, 1, MPI_INT, node+1, TAG, MPI_COMM_WORLD);
//                     printf("send row %d by node %d \n",row,node+1);
                    pool[node]=1;
                    row++;
                }
            }
            //receive result from slave nodes
            for(int node=0; node < (num_nodes-1); node++){
                if(pool[node]==1){
                    if(wait[node]==0){
                        MPI_Irecv(&node_solution[node], 1, MPI_INT, node+1, TAG, MPI_COMM_WORLD,&rq[node]);
                        wait[node]=1;
                    }
    
                    MPI_Request_get_status(rq[node],&flag[node],NULL);
                    if(flag[node]){
//                         printf("rec by node %d: %d - %d \n",row,node+1,node_solution[node]);
                        total_solution += node_solution[node];
                        pool[node]=0;
                        wait[node]=0;
                    }
                }
            }
            //waiting for last
            if(row==N){
                for(int node=0; node < (num_nodes-1); node++){
                    if(pool[node]==1){
                        MPI_Wait(&rq[node],NULL);
                        total_solution += node_solution[node];
                    }
                }
            }
        }
        printf("total solution: %d \n",total_solution);
    }
    // slave nodes handle each row
    else{
        int startRow;
        while(1){
            //receive task from master
            MPI_Recv(&startRow, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            node_solution[0] = solveNQ(startRow); 
            MPI_Isend(&node_solution[0], 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD,&rq[0]);
            MPI_Wait(&rq[0],NULL); 
        }
    }
    MPI_Finalize();
    return 0; 
} 
