#include <stdio.h>

#include <stdlib.h>

#include <time.h>



#define BLOCK_SIZE 16



__global__ void isingModel(char* inputGrid, char* outputGrid, int n) {

    int bx = blockIdx.x;

    int by = blockIdx.y;

    int tx = threadIdx.x;

    int ty = threadIdx.y;



    int row = by * blockDim.y + ty;

    int col = bx * blockDim.x + tx;



    int prevRow = (row - 1 + n) % n;

    int nextRow = (row + 1) % n;

    int prevCol = (col - 1 + n) % n;

    int nextCol = (col + 1) % n;



    char sum = inputGrid[prevRow * n + col] + inputGrid[row * n + prevCol] +

               inputGrid[row * n + col] + inputGrid[nextRow * n + col] +

               inputGrid[row * n + nextCol];



    outputGrid[row * n + col] = (sum >= 0) ? '+' : '-';

}



void printGrid(char* grid, int n) {

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            printf("%c ", grid[i * n + j]);

        }

        printf("\n");

    }

}



int main() {

    int n = 10; // Size of the grid

    int k = 100;  // Number of iterations

    printf("\tV2\n");
		
    printf("Number of iterations: %d\n", k);

    // Allocate memory for the grids

    char* grid1 = (char*)malloc(n * n * sizeof(char));

    char* grid2 = (char*)malloc(n * n * sizeof(char));



    // Initialize the grid with random spins

    srand(time(NULL));

    for (int i = 0; i < n * n; i++) {

        grid1[i] = (rand() % 2 == 0) ? '+' : '-';

    }



    // Print the initial grid

    printf("Initial Grid:\n");

    printGrid(grid1, n);

    printf("\n");



    // Allocate memory on the GPU

    char* d_grid1;

    char* d_grid2;

    cudaMalloc((void**)&d_grid1, n * n * sizeof(char));

    cudaMalloc((void**)&d_grid2, n * n * sizeof(char));



    // Copy the initial grid to the GPU

    cudaMemcpy(d_grid1, grid1, n * n * sizeof(char), cudaMemcpyHostToDevice);



    // Launch the kernel for k iterations

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int iter = 0; iter < k; iter++) {

        isingModel<<<gridSize, blockSize>>>(d_grid1, d_grid2, n);

        cudaDeviceSynchronize();

        // Swap the grids

        char* temp = d_grid1;

        d_grid1 = d_grid2;

        d_grid2 = temp;

    }



    // Copy the final grid back to the CPU

    cudaMemcpy(grid1, d_grid1, n * n * sizeof(char), cudaMemcpyDeviceToHost);



    // Print the final grid

    printf("Final Grid:\n");

    printGrid(grid1, n);



    // Free memory

    free(grid1);

    free(grid2);

    cudaFree(d_grid1);

    cudaFree(d_grid2);



    return 0;

}



