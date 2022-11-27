#pragma once

#include "RcppEigen.h"
#include "gaussian.h"
#include "lasso.h"

namespace hesso {
template<typename T, typename... Ts>
Results
lassoWrapper(const T& x, Ts&&... args)
{
  size_t p = x.cols();
  hesso::Objective<hesso::Gaussian> objective{ p };
  return lasso(x, objective, std::forward<Ts>(args)...);
}
}
