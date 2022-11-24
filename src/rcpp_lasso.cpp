#include "hesso/objective.h"
#include "hesso/update_hessian.h"
#include "rcpp_lasso_wrapper.h"
#include <RcppEigen.h>

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
Rcpp::List
update_hessian(Eigen::MatrixXd& H,
               Eigen::MatrixXd& Hinv,
               const Eigen::MatrixXd& x,
               std::vector<size_t> active_set,
               std::vector<size_t> active_set_prev,
               std::vector<size_t> active_perm,
               std::vector<size_t> active_perm_prev,
               const bool verify_hessian,
               const bool verbose)
{
  const size_t p = x.cols();

  hesso::Objective<hesso::Gaussian> objective{ p };

  updateHessian(H,
                Hinv,
                x,
                objective,
                active_set,
                active_set_prev,
                active_perm,
                active_perm_prev,
                verify_hessian,
                verbose);

  return Rcpp::List::create(Rcpp::Named("H") = Rcpp::wrap(H),
                            Rcpp::Named("Hinv") = Rcpp::wrap(Hinv));
}
