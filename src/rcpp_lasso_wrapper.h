#pragma once

#include "hesso/lasso.h"
#include <RcppEigen.h>
#include <string>

template<typename T>
Rcpp::List
rcppLassoWrapper(const T& x,
                 const Eigen::VectorXd& y,
                 const Eigen::ArrayXd& lambda,
                 const Rcpp::List args)
{
  using Rcpp::as;
  using Rcpp::Named;
  using Rcpp::wrap;

  auto tol = as<double>(args["tol"]);
  auto max_it = as<int>(args["max_it"]);
  auto warm_starts = as<bool>(args["warm_starts"]);

  auto results = hesso::lassoWrapper(x, y, lambda, tol, max_it, warm_starts);

  return Rcpp::List::create(Named("beta") = wrap(results.beta),
                            Named("gaps") = wrap(results.gaps),
                            Named("passes") = wrap(results.passes));
}
