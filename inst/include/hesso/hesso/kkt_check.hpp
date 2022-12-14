#pragma once

#include <Eigen/Core>
#include <boost/dynamic_bitset.hpp>

namespace hesso {

template<typename MatrixType>
bool
checkKktConditions(boost::dynamic_bitset<>& working,
                   Eigen::VectorXd& gradient,
                   const boost::dynamic_bitset<>& check,
                   const MatrixType& x,
                   const Eigen::VectorXd& residual,
                   const double lambda)
{
  bool any_violations{ false };

  for (size_t j = check.find_first(); j != boost::dynamic_bitset<>::npos;
       j = check.find_next(j)) {

    gradient(j) = x.col(j).dot(residual);

    if (std::abs(gradient(j)) > lambda) {
      any_violations = true;
      working[j] = true; // add feature to working set
    }
  }

  return any_violations;
}

}
