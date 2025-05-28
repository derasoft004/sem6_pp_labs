#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

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

void mat_mul(vector<vector<int>>& array1, vector<vector<int>>& array2, vector<vector<int>>& result) {
    size_t n = array1.size();
    int i, j, k;
#pragma omp parallel for shared(array1, array2, result) private(i, j, k)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            int value = 0;
            for (k = 0; k < n; k++) {
                value += array1[i][k] * array2[k][j];
            }
            result[i].push_back(value);
        }
    }
}

void test_mul() {
    vector<vector<int>> array1;
    vector<vector<int>> array2;
    vector<vector<int>> result(10, vector<int>());
    generate_array(array1, 10, 0);//0 - seed
    generate_array(array2, 10, 1);//1 - different seed
    mat_mul(array1, array2, result);

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
    vector<size_t> sizes{5, 10, 25, 50, 100, 250, 500, 750, 1000, 1100};
    vector<vector<int>> times;
    int num_tests = 20;
    for (size_t i = 0; i < sizes.size(); i++) {
        //float avg_time = 0;
        times.push_back(vector<int>());
        for (size_t j = 0; j < num_tests; j++)
        {
            vector<vector<int>> array1;
            vector<vector<int>> array2;
            vector<vector<int>> result(sizes[i], vector<int>());

            generate_array(array1, sizes[i], j);//j - seed
            generate_array(array2, sizes[i], j + 1);//j+1 - different seed

            auto start = high_resolution_clock::now();
            mat_mul(array1, array2, result);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);

            //avg_time += static_cast<int>(duration.count());
            cout << sizes[i] << ": " << duration.count() << " microseconds" << endl;
            times[i].push_back(static_cast<int>(duration.count()));
        }
    }

    ofstream outfile("times_openmp.txt");
    for (const auto row : times) {
        for (const auto value : row)
        {
            outfile << value << " ";
        }
        outfile << endl;
    }
    outfile.close();
}