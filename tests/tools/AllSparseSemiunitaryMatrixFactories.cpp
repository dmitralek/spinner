#include "AllSparseSemiunitaryMatrixFactories.h"

#include "src/entities/data_structures/arma/ArmaFactories.h"
#include "src/entities/data_structures/std/StdFactories.h"
std::vector<std::shared_ptr<quantum::linear_algebra::AbstractSparseSemiunitaryFactory>>
constructAllSparseSemiunitaryMatrixFactories() {
    return {
        std::make_shared<quantum::linear_algebra::ArmaSparseSemiunitaryFactory>(),
        std::make_shared<quantum::linear_algebra::StdSparseSemiunitaryFactory>()};
}
