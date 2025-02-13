#ifndef SPINNER_STDSPARSESEMIUNITARYMATRIX_H
#define SPINNER_STDSPARSESEMIUNITARYMATRIX_H

#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <vector>

#include "src/entities/data_structures/AbstractSparseSemiunitaryMatrix.h"

namespace quantum::linear_algebra {
class StdSparseSemiunitaryMatrix: public AbstractSparseSemiunitaryMatrix {
  public:
    StdSparseSemiunitaryMatrix() = default;
    StdSparseSemiunitaryMatrix(const StdSparseSemiunitaryMatrix&) = delete;
    StdSparseSemiunitaryMatrix& operator=(const StdSparseSemiunitaryMatrix&) = delete;
    StdSparseSemiunitaryMatrix(StdSparseSemiunitaryMatrix&&) noexcept = default;
    StdSparseSemiunitaryMatrix& operator=(StdSparseSemiunitaryMatrix&&) noexcept = default;
    ~StdSparseSemiunitaryMatrix() override = default;

    std::unique_ptr<Iterator> GetNewIterator(size_t index_of_vector) const override;

    uint32_t size_rows() const override;
    uint32_t size_cols() const override;
    bool empty() const override;
    bool vempty(uint32_t index_of_vector) const override;
    void clear() override;

    void eraseExplicitZeros() override;

    bool is_zero(uint32_t i, uint32_t j) const override;
    void move_vector_from(
        uint32_t i,
        std::unique_ptr<AbstractSparseSemiunitaryMatrix>& subspace_from) override;
    void resize(uint32_t cols, uint32_t rows);

    void add_to_position(double value, uint32_t i, uint32_t j) override;
    double at(uint32_t i, uint32_t j) const override;

    void normalize() override;

    void print(std::ostream& os) const override;

  private:
    std::vector<std::map<uint32_t, double>> basis_;
    uint32_t rows_;
    // c-like pointers are necessary to avoid double-free error
    static const StdSparseSemiunitaryMatrix*
    downcast_ptr(const std::unique_ptr<AbstractSparseSemiunitaryMatrix>& ptr);
    static StdSparseSemiunitaryMatrix*
    downcast_ptr(std::unique_ptr<AbstractSparseSemiunitaryMatrix>& ptr);
};
}  // namespace quantum::linear_algebra

#endif  //SPINNER_STDSPARSESEMIUNITARYMATRIX_H
