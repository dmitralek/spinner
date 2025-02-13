#include "FactoriesList.h"

namespace quantum::linear_algebra {

std::unique_ptr<AbstractDenseVector> FactoriesList::createVector() const {
    return denseVectorFactory_->createVector();
}

std::unique_ptr<AbstractSparseSemiunitaryMatrix>
FactoriesList::createSparseSemiunitaryMatrix(uint32_t cols, uint32_t rows) const {
    return sparseSemiunitaryMatrixFactory_->createSparseSemiunitaryMatrix(cols, rows);
}

std::unique_ptr<AbstractSymmetricMatrix>
FactoriesList::createDenseSymmetricMatrix(uint32_t size) const {
    return symmetricMatrixFactory_->createDenseSymmetricMatrix(size);
}

std::unique_ptr<AbstractSymmetricMatrix>
FactoriesList::createSparseSymmetricMatrix(uint32_t size) const {
    return symmetricMatrixFactory_->createSparseSymmetricMatrix(size);
}

std::unique_ptr<AbstractDenseSemiunitaryMatrix>
FactoriesList::createDenseSemiunitaryMatrix(uint32_t cols, uint32_t rows) const {
    return symmetricMatrixFactory_->createDenseSemiunitaryMatrix(cols, rows);
}

FactoriesList::FactoriesList(
    std::shared_ptr<AbstractSymmetricMatrixFactory> symmetricMatrixFactory,
    std::shared_ptr<AbstractSparseSemiunitaryFactory> sparseMatrix,
    std::shared_ptr<AbstractDenseVectorFactory> denseVectorMatrix) {
    symmetricMatrixFactory_ = symmetricMatrixFactory;
    sparseSemiunitaryMatrixFactory_ = sparseMatrix;
    denseVectorFactory_ = denseVectorMatrix;
}
}  // namespace quantum::linear_algebra