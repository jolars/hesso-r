#pragma once

#include <Eigen/Core>
#include <vector>

namespace hesso {

struct Results
{
  const Eigen::MatrixXd beta;
  const std::vector<double> beta0;
  const std::vector<double> primals;
  const std::vector<double> duals;
  const std::vector<double> gaps;
  const std::vector<double> lambda;
  const std::vector<size_t> passes;
};

} // namespace hesso
