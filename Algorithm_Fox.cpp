#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <ctime>
#include <chrono> 
#include <cmath>

using namespace std;
using namespace std::chrono;

// Глобальные переменные
int ProcNum = 0;    // Общее число процессов
int ProcRank = 0;   // Ранг текущего процесса
int GridSize;       // Размер квадратной решетки (число процессов по одной оси)
int GridCoords[2];  // Координаты процесса в решетке
MPI_Comm GridComm;  // Коммуникатор всей решетки
MPI_Comm RowComm;   // Коммуникатор строк решетки
MPI_Comm ColComm;   // Коммуникатор столбцов решетки

// Прототипы функций
void CreateGridCommunicators();
void ProcessInitialization(double*& pAMatrix, double*& pBMatrix, double*& pCMatrix,
    double*& pAblock, double*& pBblock, double*& pCblock,
    double*& pMatrixAblock, int& Size, int& BlockSize);
void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size);
void DataDistribution(double* pAMatrix, double* pBMatrix, double* pMatrixAblock,
    double* pBblock, int Size, int BlockSize);
void ABlockCommunication(int iter, double* pAblock, double* pMatrixAblock, int BlockSize);
void BlockMultiplication(double* pAblock, double* pBblock, double* pCblock, int BlockSize);
void BblockCommunication(double* pBblock, int BlockSize);
void ParallelResultCalculation(double* pAblock, double* pMatrixAblock,
    double* pBblock, double* pCblock, int BlockSize);
void ResultCollection(double* pCMatrix, double* pCblock, int Size, int BlockSize);
void ProcessTermination(double* pAMatrix, double* pBMatrix, double* pCMatrix,
    double* pAblock, double* pBblock, double* pCblock, double* pMatrixAblock);

// Функция для красивого вывода матрицы
void PrintMatrix(double* matrix, int size, int max_rows = 6, int max_cols = 6) {
    for (int i = 0; i < min(size, max_rows); ++i) {
        for (int j = 0; j < min(size, max_cols); ++j) {
            cout.width(7);
            cout.precision(2);
            cout << fixed << matrix[i * size + j] << " ";
        }
        cout << endl;
    }
}

// Создание коммуникаторов в виде двумерной решетки и её подрешеток.
void CreateGridCommunicators() {
    int DimSize[2] = { GridSize, GridSize };
    int Periodic[2] = { 0, 0 };
    int Subdims[2];

    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);
    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);

    Subdims[0] = 0; Subdims[1] = 1;
    MPI_Cart_sub(GridComm, Subdims, &RowComm);

    Subdims[0] = 1; Subdims[1] = 0;
    MPI_Cart_sub(GridComm, Subdims, &ColComm);
}

// Инициализация параметров задачи и выделение памяти
void ProcessInitialization(double*& pAMatrix, double*& pBMatrix, double*& pCMatrix,
    double*& pAblock, double*& pBblock, double*& pCblock,
    double*& pMatrixAblock, int& Size, int& BlockSize) {

    if (ProcRank == 0) {
        do {
            cout << "\nВведите размер матриц: ";
            cin >> Size;
            if (Size % GridSize != 0)
                cout << "Размер матриц должен быть кратен размеру сетки!" << endl;
        } while (Size % GridSize != 0);
    }

    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    BlockSize = Size / GridSize;

    pAblock = new double[BlockSize * BlockSize];
    pBblock = new double[BlockSize * BlockSize];
    pCblock = new double[BlockSize * BlockSize];
    pMatrixAblock = new double[BlockSize * BlockSize];

    for (int i = 0; i < BlockSize * BlockSize; i++)
        pCblock[i] = 0.0;

    if (ProcRank == 0) {
        pAMatrix = new double[Size * Size];
        pBMatrix = new double[Size * Size];
        pCMatrix = new double[Size * Size];
        RandomDataInitialization(pAMatrix, pBMatrix, Size);

        cout << "\nМатрица A (первые 6x6):\n";
        PrintMatrix(pAMatrix, Size);

        cout << "\nМатрица B (первые 6x6):\n";
        PrintMatrix(pBMatrix, Size);
    }
}

// Инициализация матриц A и B случайными значениями
void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
    for (int i = 0; i < Size * Size; i++) {
        pAMatrix[i] = (double)(rand() % 10);
        pBMatrix[i] = (double)(rand() % 10);
    }
}

// Распределение матриц по блокам среди процессов
void DataDistribution(double* pAMatrix, double* pBMatrix, double* pMatrixAblock,
    double* pBblock, int Size, int BlockSize) {

    int numBlocks = GridSize;

    if (ProcRank == 0) {
        for (int i = 0; i < numBlocks; i++) {
            for (int j = 0; j < numBlocks; j++) {
                int coords[2] = { i, j };
                int targetRank;
                MPI_Cart_rank(GridComm, coords, &targetRank);

                if (targetRank == 0) {
                    for (int ii = 0; ii < BlockSize; ii++) {
                        for (int jj = 0; jj < BlockSize; jj++) {
                            int global_index = (i * BlockSize + ii) * Size + j * BlockSize + jj;
                            pMatrixAblock[ii * BlockSize + jj] = pAMatrix[global_index];
                            pBblock[ii * BlockSize + jj] = pBMatrix[global_index];
                        }
                    }
                }
                else {
                    double* bufferA = new double[BlockSize * BlockSize];
                    double* bufferB = new double[BlockSize * BlockSize];

                    for (int ii = 0; ii < BlockSize; ii++) {
                        for (int jj = 0; jj < BlockSize; jj++) {
                            int global_index = (i * BlockSize + ii) * Size + j * BlockSize + jj;
                            bufferA[ii * BlockSize + jj] = pAMatrix[global_index];
                            bufferB[ii * BlockSize + jj] = pBMatrix[global_index];
                        }
                    }

                    MPI_Send(bufferA, BlockSize * BlockSize, MPI_DOUBLE, targetRank, 0, MPI_COMM_WORLD);
                    MPI_Send(bufferB, BlockSize * BlockSize, MPI_DOUBLE, targetRank, 1, MPI_COMM_WORLD);

                    delete[] bufferA;
                    delete[] bufferB;
                }
            }
        }
    }
    else {
        MPI_Status status;
        MPI_Recv(pMatrixAblock, BlockSize * BlockSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(pBblock, BlockSize * BlockSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    }
}

// Рассылка блока матрицы A по строке
void ABlockCommunication(int iter, double* pAblock, double* pMatrixAblock, int BlockSize) {
    int Pivot = (GridCoords[0] + iter) % GridSize;

    if (GridCoords[1] == Pivot) {
        for (int i = 0; i < BlockSize * BlockSize; i++)
            pAblock[i] = pMatrixAblock[i];
    }

    MPI_Bcast(pAblock, BlockSize * BlockSize, MPI_DOUBLE, Pivot, RowComm);
}

// Умножение блоков матриц
void BlockMultiplication(double* pAblock, double* pBblock, double* pCblock, int BlockSize) {
    for (int i = 0; i < BlockSize; i++) {
        for (int j = 0; j < BlockSize; j++) {
            double sum = 0.0;
            for (int k = 0; k < BlockSize; k++) {
                sum += pAblock[i * BlockSize + k] * pBblock[k * BlockSize + j];
            }
            pCblock[i * BlockSize + j] += sum;
        }
    }
}

// Циклический сдвиг блока матрицы B по столбцу
void BblockCommunication(double* pBblock, int BlockSize) {
    MPI_Status status;
    int NextProc = (GridCoords[0] + 1) % GridSize;
    int PrevProc = (GridCoords[0] - 1 + GridSize) % GridSize;

    MPI_Sendrecv_replace(pBblock, BlockSize * BlockSize, MPI_DOUBLE,
        NextProc, 0, PrevProc, 0, ColComm, &status);
}

// Параллельное умножение матриц (алгоритм Фокса)
void ParallelResultCalculation(double* pAblock, double* pMatrixAblock,
    double* pBblock, double* pCblock, int BlockSize) {

    for (int iter = 0; iter < GridSize; iter++) {
        ABlockCommunication(iter, pAblock, pMatrixAblock, BlockSize);
        BlockMultiplication(pAblock, pBblock, pCblock, BlockSize);
        BblockCommunication(pBblock, BlockSize);
    }
}

// Сборка результата на процессе 0
void ResultCollection(double* pCMatrix, double* pCblock, int Size, int BlockSize) {
    int numBlocks = GridSize;

    if (ProcRank == 0) {
        for (int ii = 0; ii < BlockSize; ii++) {
            for (int jj = 0; jj < BlockSize; jj++) {
                pCMatrix[ii * Size + jj] = pCblock[ii * BlockSize + jj];
            }
        }

        for (int proc = 1; proc < ProcNum; proc++) {
            int coords[2];
            MPI_Cart_coords(GridComm, proc, 2, coords);
            double* buffer = new double[BlockSize * BlockSize];
            MPI_Status status;

            MPI_Recv(buffer, BlockSize * BlockSize, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD, &status);

            for (int ii = 0; ii < BlockSize; ii++) {
                for (int jj = 0; jj < BlockSize; jj++) {
                    int global_i = coords[0] * BlockSize + ii;
                    int global_j = coords[1] * BlockSize + jj;
                    pCMatrix[global_i * Size + global_j] = buffer[ii * BlockSize + jj];
                }
            }

            delete[] buffer;
        }
    }
    else {
        MPI_Send(pCblock, BlockSize * BlockSize, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
}

// Освобождение памяти
void ProcessTermination(double* pAMatrix, double* pBMatrix, double* pCMatrix,
    double* pAblock, double* pBblock, double* pCblock, double* pMatrixAblock) {

    if (ProcRank == 0) {
        delete[] pAMatrix;
        delete[] pBMatrix;
        delete[] pCMatrix;
    }

    delete[] pAblock;
    delete[] pBblock;
    delete[] pCblock;
    delete[] pMatrixAblock;
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");
    srand(static_cast<unsigned int>(time(0)));
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    double* pAMatrix = NULL, * pBMatrix = NULL, * pCMatrix = NULL;
    double* pAblock = NULL, * pBblock = NULL, * pCblock = NULL, * pMatrixAblock = NULL;
    int Size, BlockSize;

    setvbuf(stdout, 0, _IONBF, 0); 

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (ProcRank == 0) cout << "Параллельное умножение матриц (алгоритм Фокса)" << endl;
    GridSize = (int)sqrt((double)ProcNum);
    if (ProcNum != GridSize * GridSize) {
        if (ProcRank == 0)
            cout << "Число процессов должно быть полным квадратом!" << endl;
        MPI_Finalize();
        return 1;
    }

    // Создаем коммуникаторы решетки, строк и столбцов
    CreateGridCommunicators();

    // Инициализация данных и выделение памяти
    ProcessInitialization(pAMatrix, pBMatrix, pCMatrix,
        pAblock, pBblock, pCblock, pMatrixAblock, Size, BlockSize);
    if (ProcRank == 0) start = high_resolution_clock::now(); //запуск таймера
  

    // Распределение данных между процессами
    DataDistribution(pAMatrix, pBMatrix, pMatrixAblock, pBblock, Size, BlockSize);

    // Выполнение параллельного умножения матриц (алгоритм Фокса)
    ParallelResultCalculation(pAblock, pMatrixAblock, pBblock, pCblock, BlockSize);

    // Сборка результата на процессе 0
    ResultCollection(pCMatrix, pCblock, Size, BlockSize);

    if (ProcRank == 0){    
        // завершение замера времени перед выводом результата
        end = high_resolution_clock::now();
        auto duration_milli = duration_cast<milliseconds>(end - start).count();
        cout << "\nРезультат умножения (первые 6x6):" << endl;
        PrintMatrix(pCMatrix, Size);
        cout << "\nВремя работы программы: " << duration_milli << " мс" << endl << endl;
    }
    // Освобождение памяти
    ProcessTermination(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock, pCblock, pMatrixAblock);

    MPI_Finalize();
    return 0;
}
