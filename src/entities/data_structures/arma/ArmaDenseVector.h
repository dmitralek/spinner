#ifndef SPINNER_ARMADENSEVECTOR_H
#define SPINNER_ARMADENSEVECTOR_H

#include <armadillo>
#include <cstddef>
#include <cstdint>

#include "src/entities/data_structures/AbstractDenseVector.h"

namespace quantum::linear_algebra {
class ArmaDenseVector: public AbstractDenseVector {

  public:
    void resize(uint32_t new_size);
    void concatenate_with(const std::unique_ptr<AbstractDenseVector>& rhs) override;
    void add_identical_values(size_t number, double value) override;
    void subtract_minimum() override;
    std::unique_ptr<AbstractDenseVector> divide_and_wise_exp(double denominator) const override;
    double dot(const std::unique_ptr<AbstractDenseVector>& rhs) const override;
    std::unique_ptr<AbstractDenseVector>
    element_wise_multiplication(const std::unique_ptr<AbstractDenseVector>& rhs) const override;
    uint32_t size() const override;
    double at(uint32_t i) const override;
    void print(std::ostream& os) const override;

    arma::dvec& modifyDenseVector();
    const arma::dvec& getDenseVector();

  private:
    // c-like pointers are necessary to avoid double-free error
    static const ArmaDenseVector* downcast_ptr(const std::unique_ptr<AbstractDenseVector>& ptr);
    arma::dvec vector_;
};
}  // namespace quantum::linear_algebra
#endif  //SPINNER_ARMADENSEVECTOR_H
