#include "ConstantOperator.h"

void ConstantOperator::construct(
    lexicographic::SparseMatrix& matrix_in_lexicografical_basis,
    uint32_t index_of_vector) const {
    matrix_in_lexicografical_basis.add_to_position(constant_, index_of_vector, index_of_vector);
}

ConstantOperator::ConstantOperator(double constant) : constant_(constant) {}

std::unique_ptr<ZeroCenterTerm> ConstantOperator::clone() const {
    return std::make_unique<ConstantOperator>(constant_);
}
