// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <variant>
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

struct FoldedTensorLayout {
    MemoryConfig memory_config;
    Layout layout;
};

MemoryConfig determine_fold_reshard_memory_config(
    const MemoryConfig& input_memory_config,
    const Shape& input_shape,
    uint32_t stride_h,
    const CoreCoord& compute_grid_size);

MemoryConfig determine_fold_output_memory_config(
    const MemoryConfig& input_memory_config, uint32_t stride_h, uint32_t stride_w);

FoldedTensorLayout determine_folded_tensor_layout(
    const Shape& input_shape,
    const MemoryConfig& input_memory_config,
    uint32_t stride_h,
    uint32_t stride_w,
    const std::array<uint32_t, 4>& padding,
    const CoreCoord& compute_grid_size);

}  // namespace ttnn::operations::data_movement

namespace ttnn {

ttnn::Tensor fold(
    const ttnn::Tensor& input_tensor,
    uint32_t stride_h,
    uint32_t stride_w,
    bool use_transpose_as_fold = false,
    const std::optional<const ttnn::Shape>& output_shape = std::nullopt,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>, std::array<uint32_t, 6>> padding =
        std::array<uint32_t, 2>{0, 0},
    const std::optional<CoreRangeSet>& core_grid = std::nullopt,
    const std::optional<MemoryConfig>& override_memory_config = std::nullopt);

}  // namespace ttnn
