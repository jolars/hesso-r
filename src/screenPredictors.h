#pragma once

#include <RcppArmadillo.h>

using namespace arma;

template<typename T>
uvec
screenPredictors(const std::string screening_type,
                 const uvec& strong,
                 const uvec& ever_active,
                 const vec& residual,
                 const vec& c,
                 const vec& c_grad,
                 const T& X,
                 const vec& X_norms,
                 const vec& X_offset,
                 const vec& y,
                 const double lambda,
                 const double lambda_next,
                 const double gamma,
                 const bool standardize)
{
  vec c_pred = c + c_grad * (lambda_next - lambda);
  return (abs(c_pred) + gamma * (lambda - lambda_next) > lambda_next) ||
         ever_active;
}
