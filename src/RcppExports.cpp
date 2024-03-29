// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcppLassoDense
Rcpp::List rcppLassoDense(const Eigen::MatrixXd& x, const Eigen::VectorXd& y, const std::vector<double>& lambda, const Rcpp::List args);
RcppExport SEXP _hesso_rcppLassoDense(SEXP xSEXP, SEXP ySEXP, SEXP lambdaSEXP, SEXP argsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type args(argsSEXP);
    rcpp_result_gen = Rcpp::wrap(rcppLassoDense(x, y, lambda, args));
    return rcpp_result_gen;
END_RCPP
}
// rcppLassoSparse
Rcpp::List rcppLassoSparse(const Eigen::SparseMatrix<double>& x, const Eigen::VectorXd& y, const std::vector<double>& lambda, const Rcpp::List args);
RcppExport SEXP _hesso_rcppLassoSparse(SEXP xSEXP, SEXP ySEXP, SEXP lambdaSEXP, SEXP argsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::SparseMatrix<double>& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const std::vector<double>& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List >::type args(argsSEXP);
    rcpp_result_gen = Rcpp::wrap(rcppLassoSparse(x, y, lambda, args));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_hesso_rcppLassoDense", (DL_FUNC) &_hesso_rcppLassoDense, 4},
    {"_hesso_rcppLassoSparse", (DL_FUNC) &_hesso_rcppLassoSparse, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_hesso(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
