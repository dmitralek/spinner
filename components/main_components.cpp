#include "Task.h"
#include "tz_sorter.h"
#include "c2_symmetrizer.h"
#include <chrono>


int main() {

    std::vector<int> mults = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4,};

    auto start = std::chrono::high_resolution_clock::now();
    int cycles = 100;
    for (int i = 0; i < cycles; ++i) {
        Task T(mults);

        T = tz_sorter(T);

        T = c2_symmetrizer(T, 5);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_once = (finish - start) / cycles;
    std::cout << "Elapsed time: " << elapsed_once.count() << " s\n";

}