#pragma once

#include <algorithm>
#include <vector>

namespace hesso {

template<typename T, typename S>
bool
contains(const T& x, const S& what)
{
  return std::find(x.cbegin(), x.cend(), what) != x.end();
}

}
