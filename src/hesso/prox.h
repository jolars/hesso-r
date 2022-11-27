#pragma once

#include "math.h"
#include <algorithm>

template<typename T>
double
prox(const T x, const T lambda)
{
  return signum(x) * std::max(std::abs(x) - lambda, 0.0);
}
