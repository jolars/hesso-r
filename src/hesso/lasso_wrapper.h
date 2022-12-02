#pragma once

#include "gaussian.h"
#include "lasso.h"
#include <Eigen/Core>

namespace hesso {
template<typename MatrixType, typename... Ts>
Results
lassoWrapper(const MatrixType& x, Ts&&... args)
{
  size_t p = x.cols();
  hesso::Objective<hesso::Gaussian> objective{ p };
  return lasso(x, objective, std::forward<Ts>(args)...);
}
}
