#pragma once

#include <Eigen/Core>
#include <vector>

namespace hesso {

struct Results
{
  const Eigen::MatrixXd beta;
  const std::vector<double> gaps;
  const std::vector<size_t> passes;
};

} // namespace hesso
