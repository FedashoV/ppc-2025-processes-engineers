#pragma once

#include "task/include/task.hpp"
#include "tsyplakov_k_rectangle_integral/common/include/common.hpp"

namespace tsyplakov_k_rectangle_integral {

class TsyplakovKRectangleIntegralMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit TsyplakovKRectangleIntegralMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tsyplakov_k_rectangle_integral
