#pragma once

#include "math.hpp"
#include <algorithm>

namespace hesso {

template<typename T>
double
prox(const T x, const T lambda)
{
  return signum(x) * std::max(std::abs(x) - lambda, 0.0);
}

}

