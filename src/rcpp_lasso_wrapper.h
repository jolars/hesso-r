#pragma once

#include "hesso/gaussian.h"
#include "hesso/lasso_wrapper.h"
#include <RcppEigen.h>

template<typename T>
Rcpp::List
rcppLassoWrapper(const T& x,
                 const Eigen::VectorXd& y,
                 const std::vector<double>& lambda,
                 const Rcpp::List args)
{
  using Rcpp::as;
  using Rcpp::Named;
  using Rcpp::wrap;

  auto path_length = as<size_t>(args["path_length"]);
  auto lambda_min_ratio = as<double>(args["lambda_min_ratio"]);
  auto warm_starts = as<bool>(args["warm_starts"]);
  auto tol = as<double>(args["tol"]);
  auto max_it = as<size_t>(args["max_it"]);

  auto results = hesso::lassoWrapper(
    x, y, lambda, path_length, lambda_min_ratio, tol, max_it, warm_starts);

  return Rcpp::List::create(Named("beta") = wrap(results.beta),
                            Named("lambda") = wrap(results.lambda),
                            Named("gaps") = wrap(results.gaps),
                            Named("passes") = wrap(results.passes));
}
