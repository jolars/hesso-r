#include "hesso/objective.h"
#include "hesso/update_hessian.h"
#include "rcpp_lasso_wrapper.h"
#include <RcppEigen.h>
#include <boost/dynamic_bitset.hpp>

// [[Rcpp::export]]
Rcpp::List
rcppLassoDense(const Eigen::MatrixXd& x,
               const Eigen::VectorXd& y,
               const std::vector<double>& lambda,
               const Rcpp::List args)
{
  return rcppLassoWrapper(x, y, lambda, args);
}

// [[Rcpp::export]]
Rcpp::List
rcppLassoSparse(const Eigen::SparseMatrix<double>& x,
                const Eigen::VectorXd& y,
                const std::vector<double>& lambda,
                const Rcpp::List args)
{
  return rcppLassoWrapper(x, y, lambda, args);
}
