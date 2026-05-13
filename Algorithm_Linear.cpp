#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <ctime>
#include <chrono> 
using namespace std;
using namespace std::chrono;

// Инициализация случайных данных в матрицах A и B
void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
    for (int i = 0; i < Size * Size; i++) {
        pAMatrix[i] = (double)(rand() % 10);
        pBMatrix[i] = (double)(rand() % 10);
    }
}

// Умножение блока A на всю матрицу B, запись в блок C
void MultiplyBlock(double* Ablock, double* B, double* Cblock, int rows, int N) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += Ablock[i * N + k] * B[k * N + j];
            }
            Cblock[i * N + j] = sum;
        }
    }
}

// Вывод матрицы (как прямоугольник)
void PrintMatrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout.width(7); 
            cout.precision(2);
            cout << fixed << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}
int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");
    srand(static_cast<unsigned int>(time(0)));
    int ProcNum, ProcRank;
    int N;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    double* A = NULL, * B = NULL, * C = NULL;
    double* Ablock = NULL, * Cblock = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) cout << "Параллельное умножение матриц (Линейный алгоритм)" << endl;
    if (ProcRank == 0) {
        do {
            cout << "Введите размер квадратных матриц (кратный числу процессов " << ProcNum << "): ";
            cin >> N;
            if (N % ProcNum != 0)
                cout << "Размер должен быть кратен числу процессов\n";
        } while (N % ProcNum != 0);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int rows_per_proc = N / ProcNum;

    // Все процессы выделяют память под B
    B = (double*)malloc(N * N * sizeof(double));
    if (!B) {
        cerr << "Ошибка выделения памяти для B на процессе " << ProcRank << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (ProcRank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        C = (double*)calloc(N * N, sizeof(double)); 
        if (!A || !C) {
            cerr << "Ошибка выделения памяти на процессе 0\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        RandomDataInitialization(A, B, N);

        cout << "Матрица A:\n";
        PrintMatrix(A, min(6, N), min(6, N));
        cout << "Матрица B:\n";
        PrintMatrix(B, min(6, N), min(6, N));
        start = high_resolution_clock::now();
    }
// Выделяем память под блоки
    Ablock = (double*)malloc(rows_per_proc * N * sizeof(double));
    Cblock = (double*)malloc(rows_per_proc * N * sizeof(double));
    if (!Ablock || !Cblock) {
        cerr << "Ошибка выделения памяти под блоки на процессе " << ProcRank << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Рассылка данных
    if (ProcNum > 1) {
        MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
            Ablock, rows_per_proc * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
        MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else {
        // Если один процесс — просто копируем всё в Ablock
        for (int i = 0; i < N * N; ++i)
            Ablock[i] = A[i];
    }
    //Параллельное умножение матриц
    MultiplyBlock(Ablock, B, Cblock, rows_per_proc, N);

    //Сборка результата на процессе 0 
    if (ProcNum > 1) {
        MPI_Gather(Cblock, rows_per_proc * N, MPI_DOUBLE,
            C, rows_per_proc * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
    }
    else {
        for (int i = 0; i < N * N; ++i)
            C[i] = Cblock[i];
    }
// Вывод результата
    if (ProcRank == 0) {
        end = high_resolution_clock::now();  //остановка таймера перед выводом результата
        cout << "Умножение матриц завершено.\n";
        cout << "Результат C (первые 6x6):\n";
        PrintMatrix(C, min(6, N), min(6, N));

        free(A);
        free(B);
        free(C);
        auto duration_milli = duration_cast<milliseconds>(end - start).count();
        cout << "\n\nВремя работы программы: " << duration_milli << " мс" << endl;
    }
    free(Ablock);
    free(Cblock);
    MPI_Finalize();
    return 0;
}
