#include "common/runner/Runner.h"
#include "gtest/gtest.h"

TEST(symbols, throw_2222_isotropic_inconsistent_symmetry) {
    std::vector<int> mults = {2, 2, 2, 2};

    runner::Runner runner(mults);

    runner.TzSort();
    runner.Symmetrize(group::Group::S2, {{1, 0, 3, 2}});
    runner.Symmetrize(group::Group::S2, {{3, 2, 1, 0}});

    double J = 10;
    runner.AddSymbol("3J", 3 * J);
    runner.AddSymbol("J", J);
    runner.AddSymbol("2J", 2 * J);
    runner.AddIsotropicExchange("3J", 0, 1);
    runner.AddIsotropicExchange("J", 1, 2);
    runner.AddIsotropicExchange("2J", 2, 3);
    runner.AddIsotropicExchange("J", 3, 0);
    runner.FinalizeIsotropicInteraction();

    EXPECT_THROW(runner.BuildMatrices(), std::invalid_argument);
}

TEST(symbols, throw_2222_isotropic_accidental_symmetry) {
    std::vector<int> mults = {2, 2, 2, 2};

    runner::Runner runner(mults);

    runner.TzSort();
    runner.Symmetrize(group::Group::S2, {{1, 0, 3, 2}});
    runner.Symmetrize(group::Group::S2, {{3, 2, 1, 0}});

    double J = 10;
    runner.AddSymbol("J1", J);
    runner.AddSymbol("J2", J);
    runner.AddIsotropicExchange("J1", 0, 1);
    runner.AddIsotropicExchange("J1", 1, 2);
    runner.AddIsotropicExchange("J1", 2, 3);
    runner.AddIsotropicExchange("J2", 3, 0);
    runner.FinalizeIsotropicInteraction();

    EXPECT_THROW(runner.BuildMatrices(), std::invalid_argument);
}

TEST(symbols, throw_2222_gfactor_inconsistent_symmetry) {
    std::vector<int> mults = {2, 2, 2, 2};

    runner::Runner runner(mults);

    runner.TzSort();
    runner.Symmetrize(group::Group::S2, {{1, 0, 3, 2}});
    runner.Symmetrize(group::Group::S2, {{3, 2, 1, 0}});

    double J = 10;
    runner.AddSymbol("J", J);
    runner.AddSymbol("g1", 2.0);
    runner.AddSymbol("g2", 3.0);
    runner.AddIsotropicExchange("J", 0, 1);
    runner.AddIsotropicExchange("J", 1, 2);
    runner.AddIsotropicExchange("J", 2, 3);
    runner.AddIsotropicExchange("J", 3, 0);
    runner.AddGFactor("g1", 0);
    runner.AddGFactor("g1", 1);
    runner.AddGFactor("g1", 2);
    runner.AddGFactor("g2", 3);

    runner.FinalizeIsotropicInteraction();

    EXPECT_THROW(runner.BuildMatrices(), std::invalid_argument);
}

TEST(symbols, throw_2222_gfactor_accidental_symmetry) {
    std::vector<int> mults = {2, 2, 2, 2};

    runner::Runner runner(mults);

    runner.TzSort();
    runner.Symmetrize(group::Group::S2, {{1, 0, 3, 2}});
    runner.Symmetrize(group::Group::S2, {{3, 2, 1, 0}});

    double J = 10;
    double g = 2.0;
    runner.AddSymbol("J", J);
    runner.AddSymbol("g1", g);
    runner.AddSymbol("g2", g);
    runner.AddIsotropicExchange("J", 0, 1);
    runner.AddIsotropicExchange("J", 1, 2);
    runner.AddIsotropicExchange("J", 2, 3);
    runner.AddIsotropicExchange("J", 3, 0);
    runner.AddGFactor("g1", 0);
    runner.AddGFactor("g1", 1);
    runner.AddGFactor("g1", 2);
    runner.AddGFactor("g2", 3);

    runner.FinalizeIsotropicInteraction();

    EXPECT_THROW(runner.BuildMatrices(), std::invalid_argument);
}

// TODO: make these tests pass
//TEST(symbols, throw_2222_gfactor_all_were_not_initialized) {
//    std::vector<int> mults = {2, 2, 2, 2};
//
//    runner::Runner runner(mults);
//
//    runner.TzSort();
//
//    double J = 10;
//    double g = 2.0;
//    runner.AddSymbol("J", J);
//    runner.AddSymbol("g1", g);
//    runner.AddIsotropicExchange("J", 0, 1);
//    runner.AddIsotropicExchange("J", 1, 2);
//    runner.AddIsotropicExchange("J", 2, 3);
//    runner.AddIsotropicExchange("J", 3, 0);
//
//    runner.FinalizeIsotropicInteraction();
//
//    EXPECT_THROW(runner.BuildMatrices(), std::length_error);
//}
//
//TEST(symbols, throw_2222_gfactor_any_was_not_initialized) {
//    std::vector<int> mults = {2, 2, 2, 2};
//
//    runner::Runner runner(mults);
//
//    runner.TzSort();
//
//    double J = 10;
//    double g = 2.0;
//    runner.AddSymbol("J", J);
//    runner.AddSymbol("g1", g);
//    runner.AddIsotropicExchange("J", 0, 1);
//    runner.AddIsotropicExchange("J", 1, 2);
//    runner.AddIsotropicExchange("J", 2, 3);
//    runner.AddIsotropicExchange("J", 3, 0);
//    runner.AddGFactor("g1", 0);
//    runner.AddGFactor("g1", 1);
//    runner.AddGFactor("g1", 2);
//
//    runner.FinalizeIsotropicInteraction();
//
//    EXPECT_THROW(runner.BuildMatrices(), std::invalid_argument);
//}

TEST(symbols, throw_set_new_value_to_unchangeable_symbol) {
    symbols::Symbols symbols_(10);

    symbols_.addSymbol("Unchangeable", NAN, false, symbols::not_specified);

    EXPECT_THROW(
        symbols_.setNewValueToChangeableSymbol("Unchangeable", INFINITY),
        std::invalid_argument);
}

TEST(symbols, throw_specified_not_as_J_symbol) {
    size_t number_of_spins = 10;
    symbols::Symbols symbols_(number_of_spins);

    double gChangeable = 2;

    symbols_.addSymbol("not_specified", NAN, true, symbols::not_specified);
    symbols_.addIsotropicExchange("not_specified", 1, 3);
    symbols_.addSymbol("gChangeable", gChangeable, true, symbols::g_factor);
    EXPECT_THROW(symbols_.addIsotropicExchange("gChangeable", 2, 7), std::invalid_argument);
}

TEST(symbols, throw_specified_not_as_g_symbol) {
    size_t number_of_spins = 10;
    symbols::Symbols symbols_(number_of_spins);

    double JChangeable = 10;

    symbols_.addSymbol("not_specified", NAN, true, symbols::not_specified);
    symbols_.addGFactor("not_specified", 1);
    symbols_.addSymbol("JChangeable", JChangeable, true, symbols::J);
    EXPECT_THROW(symbols_.addGFactor("JChangeable", 2), std::invalid_argument);
}

TEST(symbols, set_new_value_to_changeable_J_g) {
    size_t number_of_spins = 10;
    symbols::Symbols symbols_(number_of_spins);

    double JChangeable = 10;
    double gChangeable = 2;

    symbols_.addSymbol("JChangeable", JChangeable, true, symbols::J);
    symbols_.addSymbol("gChangeable", gChangeable, true, symbols::g_factor);
    symbols_.addIsotropicExchange("JChangeable", 2, 7);
    for (size_t i = 0; i < number_of_spins; ++i) {
        symbols_.addGFactor("gChangeable", i);
    }
    auto shared_ptr_J = symbols_.constructIsotropicExchangeParameters();
    auto shared_ptr_g = symbols_.constructGFactorParameters();
    EXPECT_EQ(JChangeable, shared_ptr_J->operator()(2, 7));
    EXPECT_EQ(gChangeable, shared_ptr_g->operator()(7));
    symbols_.setNewValueToChangeableSymbol("JChangeable", 2 * JChangeable);
    EXPECT_EQ(JChangeable, shared_ptr_J->operator()(2, 7));
    symbols_.setNewValueToChangeableSymbol("gChangeable", 2 * gChangeable);
    EXPECT_EQ(gChangeable, shared_ptr_g->operator()(7));
    symbols_.updateIsotropicExchangeParameters();
    EXPECT_EQ(2 * JChangeable, shared_ptr_J->operator()(2, 7));
    symbols_.updateGFactorParameters();
    EXPECT_EQ(2 * gChangeable, shared_ptr_g->operator()(7));
}