#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsyplakov_k_from_all_to_one {

// Шаблон для поддержки разных типов данных
template <typename T>
using InTypeT = std::tuple<std::vector<T>, int>;  // Массив данных и root

template <typename T>
using OutTypeT = std::vector<T>;  // На выходе получаем все данные

using TestType = std::tuple<std::vector<int>, int, std::string>;  // Параметры теста
template <typename T>
using BaseTaskT = ppc::task::Task<InTypeT<T>, OutTypeT<T>>;

}  // namespace tsyplakov_k_from_all_to_one
