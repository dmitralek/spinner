#include "gtest/gtest.h"
#include "common/Runner.h"
#include "common/Logger.h"

size_t number_of_vectors(const Space& space) {
    size_t acc = 0;
    for (const auto& subspace : space.blocks) {
        acc += subspace.basis.size();
    }
    return acc;
}

bool orthogonality_of_basis(const Space& space) {
    std::vector<DecompositionMap> unitary_matrix;
    for (const auto& subspace : space.blocks) {
        unitary_matrix.insert(unitary_matrix.end(), subspace.basis.begin(), subspace.basis.end());
    }
    for (size_t i = 0; i < unitary_matrix.size(); ++i) {
        for (size_t j = i + 1; j < unitary_matrix.size(); ++j) {
            double accumulator = 0;
            for (const auto& p : unitary_matrix[i]) {
                if (unitary_matrix[j].find(p.first) != unitary_matrix[j].end()) {
                    accumulator += p.second * unitary_matrix[j][p.first];
                }
            }
            if (accumulator != 0) {
                return false;
            }
        }
    }
    return true;
}

TEST(symmetrizer, 4444_S2) {
    std::vector<int> mults = {4, 4, 4, 4};

    runner::Runner runner(mults);
    runner.Symmetrize(group::S2, {{1, 0, 3, 2}});
    EXPECT_EQ(runner.getTotalSpaceSize(), number_of_vectors(runner.getSpace()));
    EXPECT_TRUE(orthogonality_of_basis(runner.getSpace()))
    << "Vectors are not orthogonal";
}

TEST(symmetrizer, 4444_doubleS2) {
    std::vector<int> mults = {4, 4, 4, 4};

    runner::Runner runner(mults);
    runner.Symmetrize(group::S2, {{1, 0, 3, 2}});
    runner.Symmetrize(group::S2, {{1, 0, 3, 2}});
    EXPECT_EQ(runner.getTotalSpaceSize(), number_of_vectors(runner.getSpace()));
    EXPECT_TRUE(orthogonality_of_basis(runner.getSpace()))
    << "Vectors are not orthogonal";
}

TEST(symmetrizer, 333_S3) {
    std::vector<int> mults = {3, 3, 3};

    runner::Runner runner(mults);
    runner.Symmetrize(group::S3, {{1, 2, 0}, {0, 2, 1}});
    EXPECT_EQ(runner.getTotalSpaceSize(), number_of_vectors(runner.getSpace()));
    EXPECT_TRUE(orthogonality_of_basis(runner.getSpace()))
    << "Vectors are not orthogonal";
}

TEST(symmetrizer, 333_doubleS3) {
    std::vector<int> mults = {3, 3, 3};

    runner::Runner runner(mults);
    runner.Symmetrize(group::S3, {{1, 2, 0}, {0, 2, 1}});
    runner.Symmetrize(group::S3, {{1, 2, 0}, {0, 2, 1}});
    EXPECT_EQ(runner.getTotalSpaceSize(), number_of_vectors(runner.getSpace()));
    EXPECT_TRUE(orthogonality_of_basis(runner.getSpace()))
    << "Vectors are not orthogonal";
}

TEST(symmetrizer, 333_doubleS3_tricky) {
    std::vector<int> mults = {3, 3, 3};

    runner::Runner runner(mults);
    runner.Symmetrize(group::S3, {{1, 2, 0}, {0, 2, 1}});
    runner.Symmetrize(group::S3, {{2, 0, 1}, {1, 0, 2}});
    EXPECT_EQ(runner.getTotalSpaceSize(), number_of_vectors(runner.getSpace()));
    EXPECT_TRUE(orthogonality_of_basis(runner.getSpace()))
    << "Vectors are not orthogonal";
}

TEST(symmetrizer, 333333_S3) {
    std::vector<int> mults = {3, 3, 3, 3, 3, 3};

    runner::Runner runner(mults);
    runner.Symmetrize(group::S3, {{1, 2, 0, 4, 5, 3}, {0, 2, 1, 3, 5, 4}});
    EXPECT_EQ(runner.getTotalSpaceSize(), number_of_vectors(runner.getSpace()));
    EXPECT_TRUE(orthogonality_of_basis(runner.getSpace()))
    << "Vectors are not orthogonal";
}

TEST(symmetrizer, 333333_S3xS2) {
    std::vector<int> mults = {3, 3, 3, 3, 3, 3};

    runner::Runner runner(mults);
    runner.Symmetrize(group::S3, {{1, 2, 0, 4, 5, 3}, {0, 2, 1, 3, 5, 4}});
    runner.Symmetrize(group::S2, {{3, 4, 5, 0, 1, 2}});

    EXPECT_EQ(runner.getTotalSpaceSize(), number_of_vectors(runner.getSpace()));
    EXPECT_TRUE(orthogonality_of_basis(runner.getSpace()))
    << "Vectors are not orthogonal";
}

TEST(symmetrizer, 333333_S2xS3) {
    std::vector<int> mults = {3, 3, 3, 3, 3, 3};

    runner::Runner runner(mults);
    runner.Symmetrize(group::S2, {{3, 4, 5, 0, 1, 2}});
    runner.Symmetrize(group::S3, {{1, 2, 0, 4, 5, 3}, {0, 2, 1, 3, 5, 4}});

    EXPECT_EQ(runner.getTotalSpaceSize(), number_of_vectors(runner.getSpace()));
    EXPECT_TRUE(orthogonality_of_basis(runner.getSpace()))
    << "Vectors are not orthogonal";
}

TEST(symmetrizer, 333333333_S3xS3) {
    std::vector<int> mults = {3, 3, 3, 3, 3, 3, 3, 3, 3};

    runner::Runner runner(mults);
    runner.Symmetrize(group::S3, {{1, 2, 0, 4, 5, 3, 7, 8, 6}, {0, 2, 1, 3, 5, 4, 6, 8, 7}});
    runner.Symmetrize(group::S3, {{3, 4, 5, 6, 7, 8, 0, 1, 2}, {0, 1, 2, 6, 7, 8, 3, 4, 5}});
    EXPECT_EQ(runner.getTotalSpaceSize(), number_of_vectors(runner.getSpace()));
    EXPECT_TRUE(orthogonality_of_basis(runner.getSpace()))
    << "Vectors are not orthogonal";
}
