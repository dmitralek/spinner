#include "SzSzTwoCenterTerm.h"

#include <utility>

namespace model::operators {

SzSzTwoCenterTerm::SzSzTwoCenterTerm(
    lexicographic::IndexConverter converter,
    std::shared_ptr<const TwoDNumericalParameters<double>> parameters) :
    converter_(std::move(converter)),
    coefficients_(std::move(parameters)) {}

SzSzTwoCenterTerm::SzSzTwoCenterTerm(
    lexicographic::IndexConverter converter,
    std::shared_ptr<const TwoDNumericalParameters<double>> parameters,
    double prefactor) :
    converter_(std::move(converter)),
    coefficients_(std::move(parameters)),
    prefactor_(prefactor) {}

void SzSzTwoCenterTerm::construct(
    std::unique_ptr<quantum::linear_algebra::AbstractSymmetricMatrix>&
        matrix_in_lexicografical_basis,
    uint32_t index_of_vector,
    uint32_t center_a,
    uint32_t center_b) const {
    double factor = prefactor_ * coefficients_->at(center_a, center_b);
    uint32_t projection_of_center_a =
        converter_.convert_lex_index_to_one_sz_projection(index_of_vector, center_a);
    uint32_t projection_of_center_b =
        converter_.convert_lex_index_to_one_sz_projection(index_of_vector, center_b);

    // Saz Sbz
    double diagonal_value = (projection_of_center_a - converter_.get_spins()[center_a])
        * (projection_of_center_b - converter_.get_spins()[center_b]) * factor;
    matrix_in_lexicografical_basis->add_to_position(
        diagonal_value,
        index_of_vector,
        index_of_vector);
}

std::unique_ptr<TwoCenterTerm> SzSzTwoCenterTerm::clone() const {
    return std::make_unique<SzSzTwoCenterTerm>(converter_, coefficients_);
}

}  // namespace model::operators