#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

using namespace std::chrono;
using namespace std;

void generate_array(vector<vector<int>>& array, size_t size, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(-100, 300);
    for (size_t i = 0; i < size; i++)
    {
        array.push_back(vector<int>());
        for (size_t j = 0; j < size; j++)
        {
            array[i].push_back(distrib(gen));
        }
    }
}

__global__ void matrixMultiplyKernel(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void mat_mul_cuda(const std::vector<std::vector<int>>& array1,
    const std::vector<std::vector<int>>& array2,
    std::vector<std::vector<int>>& result) {
    int N = array1.size();

    int* flatA = new int[N * N];
    int* flatB = new int[N * N];
    int* flatC = new int[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            flatA[i * N + j] = array1[i][j];
            flatB[i * N + j] = array2[i][j];
        }
    }

    int* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));
    cudaMemcpy(d_A, flatA, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiplyKernel <<<numBlocks, threadsPerBlock>>> (d_A, d_B, d_C, N);
    cudaMemcpy(flatC, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    result.resize(N);
    for (int i = 0; i < N; ++i) {
        result[i].resize(N);
        for (int j = 0; j < N; ++j) {
            result[i][j] = flatC[i * N + j];
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] flatA;
    delete[] flatB;
    delete[] flatC;
}

void test_mul() {
    vector<vector<int>> array1;
    vector<vector<int>> array2;
    vector<vector<int>> result;
    generate_array(array1, 10, 0);//0 - seed
    generate_array(array2, 10, 1);//1 - different seed
    mat_mul_cuda(array1, array2, result);

    ofstream outfile1("array1.txt");
    for (const auto row : array1) {
        for (const auto value : row)
        {
            outfile1 << value << " ";
        }
        outfile1 << endl;
    }
    outfile1.close();

    ofstream outfile2("array2.txt");
    for (const auto row : array2) {
        for (const auto value : row)
        {
            outfile2 << value << " ";
        }
        outfile2 << endl;
    }
    outfile2.close();

    ofstream outfile3("result.txt");
    for (const auto row : result) {
        for (const auto value : row)
        {
            outfile3 << value << " ";
        }
        outfile3 << endl;
    }
    outfile3.close();
}

int main()
{
    //test_mul();
    vector<size_t> sizes{5, 10, 25, 50, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000};
    //vector<size_t> sizes{ 5, 10 };
    vector<vector<int>> times;
    int num_tests = 10;
    for (size_t i = 0; i < sizes.size(); i++) {
        //float avg_time = 0;
        times.push_back(vector<int>());
        for (size_t j = 0; j < num_tests; j++)
        {
            vector<vector<int>> array1;
            vector<vector<int>> array2;
            vector<vector<int>> result;

            generate_array(array1, sizes[i], j);//j - seed
            generate_array(array2, sizes[i], j + 1);//j+1 - different seed

            auto start = high_resolution_clock::now();
            mat_mul_cuda(array1, array2, result);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);

            //avg_time += static_cast<int>(duration.count());
            cout << sizes[i] << ": " << duration.count() << " microseconds" << endl;
            times[i].push_back(static_cast<int>(duration.count()));
        }
    }

    ofstream outfile("times_cuda2.txt");
    for (const auto row : times) {
        for (const auto value : row)
        {
            outfile << value << " ";
        }
        outfile << endl;
    }
    outfile.close();
}