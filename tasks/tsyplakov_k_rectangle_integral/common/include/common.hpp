#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsyplakov_k_rectangle_integral {

using InType = std::vector<double>;
using OutType = double;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tsyplakov_k_rectangle_integral
