import matplotlib.pyplot as plt


def read_stats(file_path):
    sizes = []
    times = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            parts = line.split('|')
            size_part = parts[0].split(':')[1].strip()
            size = int(size_part.split('x')[0])
            time_part = parts[1].split(':')[1].strip().split()[0]
            time = float(time_part)
            sizes.append(size)
            times.append(time)
    return sizes, times


def main():
    sizes, times = read_stats('matrix_stats.txt')

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, 'bo-')

    plt.title('Зависимость времени выполнения от размера матрицы')
    plt.xlabel('Размер матрицы')
    plt.ylabel('Время выполнения')
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()