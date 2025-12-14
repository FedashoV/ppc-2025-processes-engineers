#include "tsyplakov_k_from_all_to_one/seq/include/ops_seq.hpp"

namespace tsyplakov_k_from_all_to_one {

// =======================================================
// Конструктор
// =======================================================
TsyplakovKFromAllToOneSEQ::TsyplakovKFromAllToOneSEQ(const InTypeSEQ &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
}

// =======================================================
// Validation
// =======================================================
bool TsyplakovKFromAllToOneSEQ::ValidationImpl() {
  auto &[data, root] = this->GetInput();
  return !data.empty() && root >= 0;
}

// =======================================================
// PreProcessing
// =======================================================
bool TsyplakovKFromAllToOneSEQ::PreProcessingImpl() {
  gathered_.clear();
  return true;
}

// =======================================================
// RunImpl
// =======================================================
bool TsyplakovKFromAllToOneSEQ::RunImpl() {
  auto &[data, root] = this->GetInput();

  // В последовательном варианте просто копируем данные
  // на «root» (обычно root = 0)
  if (root == 0) {
    this->GetOutput() = data;
  }

  return true;
}

// =======================================================
// PostProcessing
// =======================================================
bool TsyplakovKFromAllToOneSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_from_all_to_one
