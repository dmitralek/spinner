#ifndef SPINNER_UNITABSTRACTSOLVER_TESTS_H
#define SPINNER_UNITABSTRACTSOLVER_TESTS_H

#include <random>
#include <math.h>

#include "gtest/gtest.h"
#include "src/nonlinear_solver/AbstractNonlinearSolver.h"
#include "tests/tools/concreteSolverConstructors/createConcreteSolver.h"

#define RESIDUAL_ERROR_EPSILON 1e-3

template<class T>
class find_local_minima: public testing::Test {
  protected:
    find_local_minima() : solver_(createConcreteSolver<T>()) {}
    std::shared_ptr<nonlinear_solver::AbstractNonlinearSolver> const solver_;
};

TYPED_TEST_SUITE_P(find_local_minima);

TYPED_TEST_P(find_local_minima, threeD_paraboloid) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> value_dist(-10, +10);

    for (size_t _ = 0; _ < 20; ++_) {
        const double a = value_dist(rng);
        const double b = value_dist(rng);

        std::function<double(const std::vector<double>&, std::vector<double>&, bool)>
            oneStepFunction = [a,
                               b](const std::vector<double>& changeable_values,
                                  std::vector<double>& gradient,
                                  bool isGradientRequired) {
                // z = (x - a)^2 + (y - b)^2
                gradient.resize(2);
                if (isGradientRequired) {
                    gradient[0] = 2 * (changeable_values[0] - a);
                    gradient[1] = 2 * (changeable_values[1] - b);
                }
                return (changeable_values[0] - a) * (changeable_values[0] - a)
                    + (changeable_values[1] - b) * (changeable_values[1] - b);
            };

        std::vector<double> values = {0, 0};

        this->solver_->optimize(oneStepFunction, values);
        EXPECT_NEAR(values[0], a, RESIDUAL_ERROR_EPSILON);
        EXPECT_NEAR(values[1], b, RESIDUAL_ERROR_EPSILON);
    }
}

TYPED_TEST_P(find_local_minima, threeD_absolute) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> value_dist(-10, +10);

    for (size_t _ = 0; _ < 20; ++_) {
        const double a = value_dist(rng);
        const double b = value_dist(rng);

        std::function<double(const std::vector<double>&, std::vector<double>&, bool)>
            oneStepFunction = [a,
                               b](const std::vector<double>& changeable_values,
                                  std::vector<double>& gradient,
                                  bool isGradientRequired) {
                // z = |x - a| + |y - b|
                gradient.resize(2);
                if (isGradientRequired) {
                    gradient[0] = (changeable_values[0] > a) ? 1 : -1;
                    gradient[1] = (changeable_values[1] > b) ? 1 : -1;
                }
                return abs(changeable_values[0] - a) + abs(changeable_values[1] - b);
            };

        std::vector<double> values = {0, 0};

        this->solver_->optimize(oneStepFunction, values);
        EXPECT_NEAR(values[0], a, RESIDUAL_ERROR_EPSILON);
        EXPECT_NEAR(values[1], b, RESIDUAL_ERROR_EPSILON);
    }
}

TYPED_TEST_P(find_local_minima, threeD_trigonometric) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> value_dist(-1, +1);

    for (size_t _ = 0; _ < 20; ++_) {
        const double a = value_dist(rng);
        const double b = value_dist(rng);

        std::function<double(const std::vector<double>&, std::vector<double>&, bool)>
            oneStepFunction = [a,
                               b](const std::vector<double>& changeable_values,
                                  std::vector<double>& gradient,
                                  bool isGradientRequired) {
                // z = sin((x - a)^2 + (y - b)^2)
                gradient.resize(2);
                if (isGradientRequired) {
                    gradient[0] = 2*(changeable_values[0]-a)*cos(pow(changeable_values[0]-a, 2.0)
                                                                + pow(changeable_values[1]-b, 2.0));
                    gradient[1] = 2*(changeable_values[1]-b)*cos(pow(changeable_values[0]-a, 2.0)
                                                                + pow(changeable_values[1]-b, 2.0));
                }
                return sin(pow(changeable_values[0]-a, 2.0) + pow(changeable_values[1]-b, 2.0));
            };

        std::vector<double> values = {0, 0};

        this->solver_->optimize(oneStepFunction, values);
        EXPECT_NEAR(values[0], a, RESIDUAL_ERROR_EPSILON);
        EXPECT_NEAR(values[1], b, RESIDUAL_ERROR_EPSILON);
    }
}

REGISTER_TYPED_TEST_SUITE_P(find_local_minima, threeD_paraboloid, threeD_absolute, threeD_trigonometric);

#endif  //SPINNER_UNITABSTRACTSOLVER_TESTS_H
