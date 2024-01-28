#include <stdio.h>

#include <stdlib.h>

#include <time.h>



#define BLOCK_SIZE 16



__global__ void isingModel(int* grid, int* result, int n, int k) {

    int tidX = blockIdx.x * blockDim.x + threadIdx.x;

    int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    int tid = tidY * n + tidX;



    __shared__ int sharedGrid[BLOCK_SIZE + 2][BLOCK_SIZE + 2];



    // Compute the global indices for the shared memory

    int sharedX = threadIdx.x + 1;

    int sharedY = threadIdx.y + 1;



    // Compute the global indices for the grid

    int globalX = blockIdx.x * blockDim.x + threadIdx.x;

    int globalY = blockIdx.y * blockDim.y + threadIdx.y;



    // Copy data from global memory to shared memory

    sharedGrid[sharedY][sharedX] = grid[tid];



    // Handle boundary conditions

    if (threadIdx.x == 0) {

        sharedGrid[sharedY][0] = grid[globalY * n + (globalX - 1 + n) % n];

    }

    if (threadIdx.x == blockDim.x - 1) {

        sharedGrid[sharedY][BLOCK_SIZE + 1] = grid[globalY * n + (globalX + 1) % n];

    }

    if (threadIdx.y == 0) {

        sharedGrid[0][sharedX] = grid[((globalY - 1 + n) % n) * n + globalX];

    }

    if (threadIdx.y == blockDim.y - 1) {

        sharedGrid[BLOCK_SIZE + 1][sharedX] = grid[((globalY + 1) % n) * n + globalX];

    }



    __syncthreads();



    for (int iter = 0; iter < k; iter++) {

        int sum = 0;



        // Compute the sum of the neighboring spins

        sum += sharedGrid[sharedY - 1][sharedX];

        sum += sharedGrid[sharedY][sharedX - 1];

        sum += sharedGrid[sharedY][sharedX];

        sum += sharedGrid[sharedY + 1][sharedX];

        sum += sharedGrid[sharedY][sharedX + 1];



        // Update the spin based on the majority of the neighbors

        result[tid] = (sum >= 0) ? 1 : -1;



        __syncthreads();



        // Swap the pointers for the next iteration

        int* temp = grid;

        grid = result;

        result = temp;



        // Copy data from global memory to shared memory

        sharedGrid[sharedY][sharedX] = grid[tid];



        // Handle boundary conditions

        if (threadIdx.x == 0) {

            sharedGrid[sharedY][0] = grid[globalY * n + (globalX - 1 + n) % n];

        }

        if (threadIdx.x == blockDim.x - 1) {

            sharedGrid[sharedY][BLOCK_SIZE + 1] = grid[globalY * n + (globalX + 1) % n];

        }

        if (threadIdx.y == 0) {

            sharedGrid[0][sharedX] = grid[((globalY - 1 + n) % n) * n + globalX];

        }

        if (threadIdx.y == blockDim.y - 1) {

            sharedGrid[BLOCK_SIZE + 1][sharedX] = grid[((globalY + 1) % n) * n + globalX];

        }



        __syncthreads();

    }

}



int main() {

    int n = 10; // Size of the grid

    int k = 100; // Number of iterations

    printf("\tV3\n");

    printf("Number of iterations: %d\n", k);


    // Allocate memory for the grid on the host

    int* hostGrid = (int*)malloc(n * n * sizeof(int));



    // Initialize the grid with random spins (+1 or -1)

    srand(time(NULL));

    for (int i = 0; i < n * n; i++) {

        hostGrid[i] = (rand() % 2 == 0) ? 1 : -1;

    }



    // Allocate memory for the grid on the device

    int* deviceGrid;

    cudaMalloc((void**)&deviceGrid, n * n * sizeof(int));



    // Copy the grid from host to device

    cudaMemcpy(deviceGrid, hostGrid, n * n * sizeof(int), cudaMemcpyHostToDevice);



    // Allocate memory for the result grid on the device

    int* deviceResult;

    cudaMalloc((void**)&deviceResult, n * n * sizeof(int));



    // Define the grid and block dimensions

    dim3 gridSize(n / BLOCK_SIZE, n / BLOCK_SIZE);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);



    // Launch the kernel

    isingModel<<<gridSize, blockSize>>>(deviceGrid, deviceResult, n, k);



    // Copy the result grid from device to host

    cudaMemcpy(hostGrid, deviceResult, n * n * sizeof(int), cudaMemcpyDeviceToHost);



    // Print the initial grid

    printf("Initial Grid:\n");

    for (int i = 0; i < n * n; i++) {

        printf("%c ", (hostGrid[i] == 1) ? '+' : '-');

        if ((i + 1) % n == 0) {

            printf("\n");

        }

    }



    // Print the result grid

    printf("Result Grid:\n");

    for (int i = 0; i < n * n; i++) {

        printf("%c ", (hostGrid[i] == 1) ? '+' : '-');

        if ((i + 1) % n == 0) {

            printf("\n");

        }

    }



    // Free memory on the device

    cudaFree(deviceGrid);

    cudaFree(deviceResult);



    // Free memory on the host

    free(hostGrid);



    return 0;

}


