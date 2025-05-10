#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <sstream>
#include <chrono>

using namespace std;

void write_to_file(const vector<vector<int>>& matrix, const string& path) {
    std::ofstream out(path);
    for (const auto& row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j != row.size() - 1) {
                out << ",";
            }
        }
        out << "\n";
    }
}


vector<vector<int>> generate(size_t size) {
    auto engine = mt19937(std::time(nullptr));

    vector<vector<int>> matrix(size, vector<int>(size));
    uniform_int_distribution<int> dist(0, 10);
    for (auto& row : matrix) {
        for (int& elem : row) {
            elem = dist(engine);
        }
    }
    return matrix;
}

vector<vector<int>> read_from_file(const string& path) {
    std::ifstream in(path);
    vector<vector<int>> matrix;
    string line;

    while (getline(in, line)) {
        istringstream iss(line);
        vector<int> row;
        string value;
        while (getline(iss, value, ',')) {
            row.push_back(stoi(value));
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    return matrix;
}

vector<vector<int>> mul_matrix(const vector<vector<int>>& matrix1, const vector<vector<int>>& matrix2) {
    size_t size = matrix1.size();
    vector<vector<int>> result(size, vector<int>(size, 0));
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            for (size_t k = 0; k < size; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}

int main() {
    vector<int> counts = {3, 5, 10, 20, 50, 100, 300, 500, 1000};
    for (const auto& count : counts) {
        for (int i = 1; i < 3; ++i) {
            vector<vector<int>> matrix = generate(count);
            string path = "../matrix_" + to_string(i) + "_size_" + to_string(count) + ".csv";
            write_to_file(matrix, path);
        }
    }

    vector<double> times(counts.size(), 0.0);
    for (size_t i = 0; i < counts.size(); ++i) {
        int count = counts[i];
        string path_1 = "../matrix_1_size_" + to_string(count) + ".csv";
        string path_2 = "../matrix_2_size_" + to_string(count) + ".csv";
        string result_path = "../result_size_" + to_string(count) + ".csv";

        auto start_time = chrono::steady_clock::now();
        vector<vector<int>> matrix_1 = read_from_file(path_1);
        vector<vector<int>> matrix_2 = read_from_file(path_2);
        vector<vector<int>> result = mul_matrix(matrix_1, matrix_2);
        write_to_file(result, result_path);
        auto end_time = chrono::steady_clock::now();

        times[i] = chrono::duration<double, milli>(end_time - start_time).count();
    }

    std::ofstream out("../matrix_stats.txt");
    for (size_t i = 0; i < counts.size(); ++i) {
        out << "размер матрицы: " << counts[i] << "x" << counts[i]
            << " | время рассчета: " << times[i] << endl;
    }

    return 0;
}