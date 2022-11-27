#include "hesso/objective.h"
#include "hesso/update_hessian.h"
#include "rcpp_lasso_wrapper.h"
#include <RcppEigen.h>
#include <boost/dynamic_bitset.hpp>

// [[Rcpp::export]]
Rcpp::List
rcppLassoDense(const Eigen::MatrixXd& x,
               const Eigen::VectorXd& y,
               const Eigen::ArrayXd& lambda,
               const Rcpp::List args)
{
  return rcppLassoWrapper(x, y, lambda, args);
}


// [[Rcpp::export]]
void
test_bitset()
{
  boost::dynamic_bitset<> working(5);

  working.set(1);
  working.set(4);

  for (size_t j = working.find_first(); j != boost::dynamic_bitset<>::npos;
       j = working.find_next(j)) {
    Rcpp::Rcout << j << " ";
  }
  Rcpp::Rcout << std::endl;
}
