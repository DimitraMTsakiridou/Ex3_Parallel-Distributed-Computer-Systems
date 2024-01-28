#include <stdio.h>

#include <stdlib.h>

#include <time.h>



void initializeGrid(int n, int* grid) {

    srand(time(NULL));

    for (int i = 0; i < n * n; i++) {

        grid[i] = (rand() % 2 == 0) ? 1 : -1;

    }

}



void evolve(int n, int* currentGrid, int* nextGrid) {

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            int sum = currentGrid[((i - 1 + n) % n) * n + j] +

                      currentGrid[i * n + ((j - 1 + n) % n)] +

                      currentGrid[i * n + j] +

                      currentGrid[((i + 1) % n) * n + j] +

                      currentGrid[i * n + ((j + 1) % n)];



            nextGrid[i * n + j] = (sum >= 0) ? 1 : -1;

        }

    }

}



void printGrid(int n, int* grid) {

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            printf("%c ", (grid[i * n + j] == 1) ? '+' : '-');

        }

        printf("\n");

    }

}



int main() {

    int n = 10; // Size of the grid

    int k = 100;  // Number of iterations



    int* currentGrid = (int*)malloc(n * n * sizeof(int));

    int* nextGrid = (int*)malloc(n * n * sizeof(int));



    initializeGrid(n, currentGrid);


    printf("\tV0\n");

    printf("Number of iterations: %d\n", k);

    printf("Initial state:\n");

    printGrid(n, currentGrid);



    for (int iteration = 1; iteration <= k; iteration++) {

        evolve(n, currentGrid, nextGrid);

        int* temp = currentGrid;

        currentGrid = nextGrid;

        nextGrid = temp;

    }



    printf("\nFinal state:\n");

    printGrid(n, currentGrid);



    free(currentGrid);

    free(nextGrid);



    return 0;

}


