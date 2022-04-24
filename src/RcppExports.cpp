// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// lassoPathDense
Rcpp::List lassoPathDense(arma::mat X, arma::vec y, const std::string family, arma::vec lambdas, const std::string lambda_type, const bool standardize, const std::string screening_type, const bool shuffle, const arma::uword check_frequency, const arma::uword screen_frequency, const bool hessian_warm_starts, const bool celer_use_old_dual, const bool celer_use_accel, const bool celer_prune, const bool gap_safe_active_start, const bool augment_with_gap_safe, std::string log_hessian_update_type, const arma::uword log_hessian_auto_update_freq, const arma::uword path_length, const arma::uword maxit, const double tol_gap, const double gamma, const bool store_dual_variables, const bool verify_hessian, const bool line_search, const arma::uword verbosity);
RcppExport SEXP _hesso_lassoPathDense(SEXP XSEXP, SEXP ySEXP, SEXP familySEXP, SEXP lambdasSEXP, SEXP lambda_typeSEXP, SEXP standardizeSEXP, SEXP screening_typeSEXP, SEXP shuffleSEXP, SEXP check_frequencySEXP, SEXP screen_frequencySEXP, SEXP hessian_warm_startsSEXP, SEXP celer_use_old_dualSEXP, SEXP celer_use_accelSEXP, SEXP celer_pruneSEXP, SEXP gap_safe_active_startSEXP, SEXP augment_with_gap_safeSEXP, SEXP log_hessian_update_typeSEXP, SEXP log_hessian_auto_update_freqSEXP, SEXP path_lengthSEXP, SEXP maxitSEXP, SEXP tol_gapSEXP, SEXP gammaSEXP, SEXP store_dual_variablesSEXP, SEXP verify_hessianSEXP, SEXP line_searchSEXP, SEXP verbositySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< const std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambdas(lambdasSEXP);
    Rcpp::traits::input_parameter< const std::string >::type lambda_type(lambda_typeSEXP);
    Rcpp::traits::input_parameter< const bool >::type standardize(standardizeSEXP);
    Rcpp::traits::input_parameter< const std::string >::type screening_type(screening_typeSEXP);
    Rcpp::traits::input_parameter< const bool >::type shuffle(shuffleSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type check_frequency(check_frequencySEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type screen_frequency(screen_frequencySEXP);
    Rcpp::traits::input_parameter< const bool >::type hessian_warm_starts(hessian_warm_startsSEXP);
    Rcpp::traits::input_parameter< const bool >::type celer_use_old_dual(celer_use_old_dualSEXP);
    Rcpp::traits::input_parameter< const bool >::type celer_use_accel(celer_use_accelSEXP);
    Rcpp::traits::input_parameter< const bool >::type celer_prune(celer_pruneSEXP);
    Rcpp::traits::input_parameter< const bool >::type gap_safe_active_start(gap_safe_active_startSEXP);
    Rcpp::traits::input_parameter< const bool >::type augment_with_gap_safe(augment_with_gap_safeSEXP);
    Rcpp::traits::input_parameter< std::string >::type log_hessian_update_type(log_hessian_update_typeSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type log_hessian_auto_update_freq(log_hessian_auto_update_freqSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type path_length(path_lengthSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const double >::type tol_gap(tol_gapSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< const bool >::type store_dual_variables(store_dual_variablesSEXP);
    Rcpp::traits::input_parameter< const bool >::type verify_hessian(verify_hessianSEXP);
    Rcpp::traits::input_parameter< const bool >::type line_search(line_searchSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type verbosity(verbositySEXP);
    rcpp_result_gen = Rcpp::wrap(lassoPathDense(X, y, family, lambdas, lambda_type, standardize, screening_type, shuffle, check_frequency, screen_frequency, hessian_warm_starts, celer_use_old_dual, celer_use_accel, celer_prune, gap_safe_active_start, augment_with_gap_safe, log_hessian_update_type, log_hessian_auto_update_freq, path_length, maxit, tol_gap, gamma, store_dual_variables, verify_hessian, line_search, verbosity));
    return rcpp_result_gen;
END_RCPP
}
// lassoPathSparse
Rcpp::List lassoPathSparse(arma::sp_mat X, arma::vec y, const std::string family, arma::vec lambdas, const std::string lambda_type, const bool standardize, const std::string screening_type, const bool shuffle, const arma::uword check_frequency, const arma::uword screen_frequency, const bool hessian_warm_starts, const bool celer_use_old_dual, const bool celer_use_accel, const bool celer_prune, const bool gap_safe_active_start, const bool augment_with_gap_safe, std::string log_hessian_update_type, const arma::uword log_hessian_auto_update_freq, const arma::uword path_length, const arma::uword maxit, const double tol_gap, const double gamma, const bool store_dual_variables, const bool verify_hessian, const bool line_search, const arma::uword verbosity);
RcppExport SEXP _hesso_lassoPathSparse(SEXP XSEXP, SEXP ySEXP, SEXP familySEXP, SEXP lambdasSEXP, SEXP lambda_typeSEXP, SEXP standardizeSEXP, SEXP screening_typeSEXP, SEXP shuffleSEXP, SEXP check_frequencySEXP, SEXP screen_frequencySEXP, SEXP hessian_warm_startsSEXP, SEXP celer_use_old_dualSEXP, SEXP celer_use_accelSEXP, SEXP celer_pruneSEXP, SEXP gap_safe_active_startSEXP, SEXP augment_with_gap_safeSEXP, SEXP log_hessian_update_typeSEXP, SEXP log_hessian_auto_update_freqSEXP, SEXP path_lengthSEXP, SEXP maxitSEXP, SEXP tol_gapSEXP, SEXP gammaSEXP, SEXP store_dual_variablesSEXP, SEXP verify_hessianSEXP, SEXP line_searchSEXP, SEXP verbositySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::sp_mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< const std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambdas(lambdasSEXP);
    Rcpp::traits::input_parameter< const std::string >::type lambda_type(lambda_typeSEXP);
    Rcpp::traits::input_parameter< const bool >::type standardize(standardizeSEXP);
    Rcpp::traits::input_parameter< const std::string >::type screening_type(screening_typeSEXP);
    Rcpp::traits::input_parameter< const bool >::type shuffle(shuffleSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type check_frequency(check_frequencySEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type screen_frequency(screen_frequencySEXP);
    Rcpp::traits::input_parameter< const bool >::type hessian_warm_starts(hessian_warm_startsSEXP);
    Rcpp::traits::input_parameter< const bool >::type celer_use_old_dual(celer_use_old_dualSEXP);
    Rcpp::traits::input_parameter< const bool >::type celer_use_accel(celer_use_accelSEXP);
    Rcpp::traits::input_parameter< const bool >::type celer_prune(celer_pruneSEXP);
    Rcpp::traits::input_parameter< const bool >::type gap_safe_active_start(gap_safe_active_startSEXP);
    Rcpp::traits::input_parameter< const bool >::type augment_with_gap_safe(augment_with_gap_safeSEXP);
    Rcpp::traits::input_parameter< std::string >::type log_hessian_update_type(log_hessian_update_typeSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type log_hessian_auto_update_freq(log_hessian_auto_update_freqSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type path_length(path_lengthSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const double >::type tol_gap(tol_gapSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< const bool >::type store_dual_variables(store_dual_variablesSEXP);
    Rcpp::traits::input_parameter< const bool >::type verify_hessian(verify_hessianSEXP);
    Rcpp::traits::input_parameter< const bool >::type line_search(line_searchSEXP);
    Rcpp::traits::input_parameter< const arma::uword >::type verbosity(verbositySEXP);
    rcpp_result_gen = Rcpp::wrap(lassoPathSparse(X, y, family, lambdas, lambda_type, standardize, screening_type, shuffle, check_frequency, screen_frequency, hessian_warm_starts, celer_use_old_dual, celer_use_accel, celer_prune, gap_safe_active_start, augment_with_gap_safe, log_hessian_update_type, log_hessian_auto_update_freq, path_length, maxit, tol_gap, gamma, store_dual_variables, verify_hessian, line_search, verbosity));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_hesso_lassoPathDense", (DL_FUNC) &_hesso_lassoPathDense, 26},
    {"_hesso_lassoPathSparse", (DL_FUNC) &_hesso_lassoPathSparse, 26},
    {NULL, NULL, 0}
};

RcppExport void R_init_hesso(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
