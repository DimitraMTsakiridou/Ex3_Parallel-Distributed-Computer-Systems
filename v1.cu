#include <stdio.h>

#include <stdlib.h>

#include <time.h>



void simulateIsingModel(int n, int k) {

    // Allocate memory for the grid

    int* grid = (int*)malloc(n * n * sizeof(int));

    int* newGrid = (int*)malloc(n * n * sizeof(int));



    // Initialize the grid with random spins (+1 or -1)

    srand(time(NULL));

    for (int i = 0; i < n * n; i++) {

        grid[i] = (rand() % 2) * 2 - 1;

    }



    // Print the initial grid

    printf("Initial Grid:\n");

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            printf("%c ", (grid[i * n + j] == 1) ? '+' : '-');

        }

        printf("\n");

    }

    printf("\n");



    // Simulate the Ising model for k iterations

    for (int iter = 0; iter < k; iter++) {

        // Update the grid

        for (int i = 0; i < n; i++) {

            for (int j = 0; j < n; j++) {

                int sum = grid[((i - 1 + n) % n) * n + j] +

                          grid[i * n + ((j - 1 + n) % n)] +

                          grid[i * n + j] +

                          grid[((i + 1) % n) * n + j] +

                          grid[i * n + ((j + 1) % n)];

                newGrid[i * n + j] = (sum > 0) ? 1 : -1;

            }

        }



        // Swap the grid pointers

        int* temp = grid;

        grid = newGrid;

        newGrid = temp;

    }



    // Print the final grid

    printf("Final Grid:\n");

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            printf("%c ", (grid[i * n + j] == 1) ? '+' : '-');

        }

        printf("\n");

    }



    // Free the memory

    free(grid);

    free(newGrid);

}



int main() {

    int n = 10; // Size of the grid

    int k = 100;  // Number of iterations

    printf("\tV1\n");

    printf("Number of iterations: %d\n", k);

    simulateIsingModel(n, k);



    return 0;

}


