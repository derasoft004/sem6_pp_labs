import numpy as np

def check_matrix_multiplication(size):
    matrix_1 = np.loadtxt(f'matrix_1_size_{size}.csv', delimiter=',')
    matrix_2 = np.loadtxt(f'matrix_2_size_{size}.csv', delimiter=',')
    mul_result = np.loadtxt(f'result_size_{size}.csv', delimiter=',')

    return np.allclose(matrix_1 @ matrix_2, mul_result)


def main():
    sizes = [3, 5, 10, 20, 50, 100, 300, 500, 1000]

    for size in sizes:
        result = check_matrix_multiplication(size)
        if result:
            print(f"Проверка пройдена для размера {size}x{size}")
        else:
            print(f"Ошибка в результате для размера {size}x{size}")


if __name__ == "__main__":
    main()
