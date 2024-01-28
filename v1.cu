#include <iostream>

#include <cstdlib>

#include <ctime>



#define BLOCK_SIZE 16



__global__ void isingModel(int* grid, int* result, int n, int k) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    

    int index = i * n + j;

    

    int left = (j == 0) ? grid[i * n + n - 1] : grid[i * n + j - 1];

    int right = (j == n - 1) ? grid[i * n] : grid[i * n + j + 1];

    int top = (i == 0) ? grid[(n - 1) * n + j] : grid[(i - 1) * n + j];

    int bottom = (i == n - 1) ? grid[j] : grid[(i + 1) * n + j];

    

    int sum = left + right + top + bottom + grid[index];

    result[index] = (sum >= 0) ? 1 : -1;

}



void printGrid(int* grid, int n) {

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            std::cout << ((grid[i * n + j] == 1) ? "+" : "-") << " ";

        }

        std::cout << std::endl;

    }

}



int main() {

    int n = 10; // Size of the grid

    int k = 100; // Number of iterations

    printf("\tV1\n");

    printf("Number of iterations: %d\n", k);

    // Allocate memory for the grid

    int* grid = new int[n * n];

    int* result = new int[n * n];

    

    // Initialize the grid with random values

    std::srand(std::time(nullptr));

    for (int i = 0; i < n * n; i++) {

        grid[i] = (std::rand() % 2 == 0) ? 1 : -1;

    }

    

    // Print the initial grid

    std::cout << "Initial Grid:" << std::endl;

    printGrid(grid, n);

    

    // Allocate memory on the GPU

    int* d_grid;

    int* d_result;

    cudaMalloc((void**)&d_grid, n * n * sizeof(int));

    cudaMalloc((void**)&d_result, n * n * sizeof(int));

    

    // Copy the grid from host to device

    cudaMemcpy(d_grid, grid, n * n * sizeof(int), cudaMemcpyHostToDevice);

    

    // Launch the kernel

    dim3 gridSize(n / BLOCK_SIZE, n / BLOCK_SIZE);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    for (int i = 0; i < k; i++) {

        isingModel<<<gridSize, blockSize>>>(d_grid, d_result, n, k);

        std::swap(d_grid, d_result);

    }

    

    // Copy the result from device to host

    cudaMemcpy(result, d_grid, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    

    // Print the result grid

    std::cout << "Final Grid:" << std::endl;

    printGrid(result, n);

    

    // Free memory on the GPU

    cudaFree(d_grid);

    cudaFree(d_result);

    

    // Free memory on the host

    delete[] grid;

    delete[] result;

    

    return 0;

}


