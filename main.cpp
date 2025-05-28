#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include "mpi.h"

using namespace std::chrono;
using namespace std;

void generate_array(vector<vector<int>>& array, size_t size, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(-100, 300);
    for (size_t i = 0; i < size; i++) {
        array.push_back(vector<int>());
        for (size_t j = 0; j < size; j++) {
            array[i].push_back(distrib(gen));
        }
    }
}

void multiply_matrix(vector<vector<int>>& array1, vector<vector<int>>& array2,
    vector<vector<int>>& result, int rank, int size) {
    const int n = array1.size();
    result.resize(n, vector<int>(n, 0));

    int rows_per_process = n / size;
    int extra_rows = n % size;
    int start_row = rank * rows_per_process + min(rank, extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            int value = 0;
            for (int k = 0; k < n; k++) {
                value += array1[i][k] * array2[k][j];
            }
            result[i][j] = value;
        }
    }
    if (rank == 0) {
        for (int src = 1; src < size; src++) {
            int src_start = src * rows_per_process + min(src, extra_rows);
            int src_end = src_start + rows_per_process + (src < extra_rows ? 1 : 0);

            for (int i = src_start; i < src_end; i++) {
                MPI_Recv(result[i].data(), n, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    else {
        for (int i = start_row; i < end_row; i++) {
            MPI_Send(result[i].data(), n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<size_t> sizes{ 5, 10, 25, 50, 100, 250, 500, 750, 1000, 1100 };
    vector<vector<int>> times;
    int num_tests = 20;

    for (size_t s = 0; s < sizes.size(); s++) {
        vector<int> local_times;

        if (rank == 0) {
            times.push_back(vector<int>());
        }

        for (int j = 0; j < num_tests; j++) {
            vector<vector<int>> array1, array2, result;

            if (rank == 0) {
                generate_array(array1, sizes[s],j);
                generate_array(array2, sizes[s], j + 1);
            }

            int n = sizes[s];
            MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                for (int i = 0; i < n; i++) {
                    MPI_Bcast(array1[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(array2[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);
                }
            }
            else {
                array1.resize(n, vector<int>(n));
                array2.resize(n, vector<int>(n));
                for (int i = 0; i < n; i++) {
                    MPI_Bcast(array1[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(array2[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = high_resolution_clock::now();

            multiply_matrix(array1, array2, result, rank, size);

            MPI_Barrier(MPI_COMM_WORLD);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);

            if (rank == 0) {
                times[s].push_back(static_cast<int>(duration.count()));
                cout << sizes[s] << " " << duration.count() << " ms" << endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}