#ifndef JULY_SPACE_H
#define JULY_SPACE_H

#include <deque>
#include <map>
#include "Subspace.h"
#include "Indexes.h"
#include <iostream>

struct Space {
    bool is_Tz_sorted = false;
    bool is_C2_symmetrized = false;

    std::deque<Subspace> blocks;

    explicit Space(const Spaces::Indexes& indexes);

    void print();

};


#endif //JULY_SPACE_H
