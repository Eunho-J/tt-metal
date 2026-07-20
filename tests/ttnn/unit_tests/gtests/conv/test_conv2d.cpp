// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <tuple>
#include <vector>
#include <tt_stl/small_vector.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/conv/conv1d/conv1d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv_types.hpp"
#include "ttnn/operations/data_movement/fold/fold.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/operations/core/core.hpp"
#include "common_test_utils.hpp"

namespace ttnn::operations::conv::conv2d::test {

struct Conv2DParam {
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t batch_size;
    uint32_t input_height;
    uint32_t input_width;
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 2> padding;
};

class Conv2DFixture : public ::testing::Test, public testing::WithParamInterface<Conv2DParam> {};

TEST(Conv2DSlicePlannerTest, TileHeightSliceUsesEffectiveRoundedWidth) {
    const auto slice_config =
        op_slicing::Op2DSliceConfig{.slice_type = op_slicing::Op2DSliceConfig::SliceType::DRAM_HEIGHT, .num_slices = 2};
    const auto output_range = op_slicing::get_output_slice_range(64, Layout::TILE, slice_config, 0);
    ASSERT_EQ(output_range.start, 0);
    ASSERT_EQ(output_range.end, 32);

    const auto plan = determine_conv2d_slice_plan(
        {64, 50}, {output_range.start, 0}, {output_range.end, 50}, {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, Layout::TILE);

    EXPECT_EQ(plan.input_start, (std::array<uint32_t, 2>{0, 0}));
    EXPECT_EQ(plan.input_end, (std::array<uint32_t, 2>{33, 50}));
    EXPECT_EQ(plan.padding, (std::array<uint32_t, 4>{1, 0, 1, 15}));
    EXPECT_EQ(plan.output_shape, (std::array<uint32_t, 2>{32, 64}));
}

TEST(Conv2DSlicePlannerTest, PreservesPaddingRequiredBeforeTheFinalSlice) {
    const auto plan =
        determine_conv2d_slice_plan({4, 32}, {0, 0}, {3, 32}, {3, 3}, {1, 1}, {0, 3, 1, 1}, {1, 1}, Layout::ROW_MAJOR);

    EXPECT_EQ(plan.input_start, (std::array<uint32_t, 2>{0, 0}));
    EXPECT_EQ(plan.input_end, (std::array<uint32_t, 2>{4, 32}));
    EXPECT_EQ(plan.padding, (std::array<uint32_t, 4>{0, 1, 1, 1}));
    EXPECT_EQ(plan.output_shape, (std::array<uint32_t, 2>{3, 32}));
}

TEST(Conv2DSlicePlannerTest, TileWidthSliceCountUsesTileUnits) {
    EXPECT_EQ(op_slicing::get_max_num_slices(64, Layout::TILE, op_slicing::Op2DSliceConfig::SliceType::DRAM_WIDTH), 2);
    EXPECT_THROW(
        op_slicing::validate_slice_config(
            64,
            Layout::TILE,
            op_slicing::Op2DSliceConfig{
                .slice_type = op_slicing::Op2DSliceConfig::SliceType::DRAM_WIDTH, .num_slices = 3}),
        std::runtime_error);

    const auto slice_config =
        op_slicing::Op2DSliceConfig{.slice_type = op_slicing::Op2DSliceConfig::SliceType::DRAM_WIDTH, .num_slices = 3};
    EXPECT_EQ(op_slicing::get_output_slice_range(65, Layout::TILE, slice_config, 0).end, 32);
    EXPECT_EQ(op_slicing::get_output_slice_range(65, Layout::TILE, slice_config, 1).end, 64);
    EXPECT_EQ(op_slicing::get_output_slice_range(65, Layout::TILE, slice_config, 2).end, 65);
}

TEST(Conv2DSlicePlannerTest, RejectsZeroSliceCountBeforeComputingRange) {
    const auto slice_config =
        op_slicing::Op2DSliceConfig{.slice_type = op_slicing::Op2DSliceConfig::SliceType::DRAM_WIDTH, .num_slices = 0};

    EXPECT_THROW(op_slicing::validate_slice_config(64, Layout::TILE, slice_config), std::runtime_error);
    EXPECT_THROW((void)op_slicing::get_output_slice_range(64, Layout::TILE, slice_config, 0), std::runtime_error);
}

TEST(Conv2DFoldPlannerTest, PreservesPhysicalChannelsInHeightShardedFold) {
    const auto input_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}};
    const auto input_memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{input_grid, {64, 8}, ShardOrientation::ROW_MAJOR}};

    const auto folded_layout = data_movement::determine_folded_tensor_layout(
        Shape({1, 8, 8, 3}), input_memory_config, 2, 2, {0, 0, 0, 0}, CoreCoord{13, 10});

    EXPECT_EQ(folded_layout.layout, Layout::ROW_MAJOR);
    EXPECT_EQ(folded_layout.memory_config.memory_layout(), TensorMemoryLayout::HEIGHT_SHARDED);
    ASSERT_TRUE(folded_layout.memory_config.shard_spec().has_value());
    EXPECT_EQ(folded_layout.memory_config.shard_spec()->grid, input_grid);
    EXPECT_EQ(folded_layout.memory_config.shard_spec()->shape, (std::array<uint32_t, 2>{16, 32}));

    const auto folded_spec = TensorSpec(
        Shape({1, 4, 4, 12}), TensorLayout(DataType::BFLOAT16, Layout::ROW_MAJOR, folded_layout.memory_config));
    EXPECT_EQ(folded_spec.padded_shape()[-1], 32);
}

TEST(Conv2DFoldPlannerTest, MatchesChannelPadReshardForOverprovisionedHeightShard) {
    const auto input_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 0}}};
    const auto input_memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{input_grid, {64, 8}, ShardOrientation::ROW_MAJOR}};

    const auto folded_layout = data_movement::determine_folded_tensor_layout(
        Shape({1, 8, 8, 3}), input_memory_config, 2, 2, {0, 0, 0, 0}, CoreCoord{13, 10});

    EXPECT_EQ(folded_layout.layout, Layout::ROW_MAJOR);
    ASSERT_TRUE(folded_layout.memory_config.shard_spec().has_value());
    EXPECT_EQ(folded_layout.memory_config.shard_spec()->grid, input_grid);
    EXPECT_EQ(folded_layout.memory_config.shard_spec()->shape, (std::array<uint32_t, 2>{8, 32}));
}

TEST(Conv2DFoldPlannerTest, MatchesHaloShardHeightForBatchedPadding) {
    const auto input_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}};
    const auto input_memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{input_grid, {1, 8}, ShardOrientation::ROW_MAJOR}};

    const auto folded_layout = data_movement::determine_folded_tensor_layout(
        Shape({3, 1, 1, 8}), input_memory_config, 2, 1, {0, 1, 0, 0}, CoreCoord{13, 10});

    EXPECT_EQ(folded_layout.layout, Layout::ROW_MAJOR);
    ASSERT_TRUE(folded_layout.memory_config.shard_spec().has_value());
    EXPECT_EQ(folded_layout.memory_config.shard_spec()->grid, input_grid);
    EXPECT_EQ(folded_layout.memory_config.shard_spec()->shape, (std::array<uint32_t, 2>{1, 16}));
}

TEST(Conv2DFoldPlannerTest, RejectsUnsupportedShardedLayout) {
    const auto input_memory_config = MemoryConfig{
        TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{
            CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 0}}}, {32, 32}, ShardOrientation::ROW_MAJOR}};

    EXPECT_THROW(
        (void)data_movement::determine_folded_tensor_layout(
            Shape({1, 8, 8, 64}), input_memory_config, 2, 2, {0, 0, 0, 0}, CoreCoord{13, 10}),
        std::runtime_error);

    const auto missing_shard_spec = MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1};
    EXPECT_THROW(
        (void)data_movement::determine_folded_tensor_layout(
            Shape({1, 8, 8, 64}), missing_shard_spec, 2, 2, {0, 0, 0, 0}, CoreCoord{13, 10}),
        std::runtime_error);
}

TEST(Conv1DContractTest, RejectsInputShapeMismatchedWithParametersWithoutOpeningDevice) {
    const auto input_shape = Shape{2, 17, 96};
    const auto weight_shape = Shape{96, 1, 4};
    const auto input_spec = TensorSpec(input_shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    const auto weight_spec =
        TensorSpec(weight_shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    const auto input = Tensor::from_vector(std::vector<float>(input_shape.volume()), input_spec);
    const auto weight = Tensor::from_vector(std::vector<float>(weight_shape.volume()), weight_spec);

    EXPECT_THROW(
        (void)ttnn::conv1d(
            input,
            weight,
            /*device=*/nullptr,
            /*in_channels=*/96,
            /*out_channels=*/96,
            /*batch_size=*/1,
            /*input_length=*/17,
            /*kernel_size=*/4),
        std::runtime_error);
}

TEST(Conv2DInputPlannerTest, ReshardOptimalityUsesCanonicalChannelAlignment) {
    const CoreCoord compute_grid{13, 10};

    const auto existing_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 7}}};
    const auto existing_memory_config = MemoryConfig{
        TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{existing_grid, {32, 32}, ShardOrientation::ROW_MAJOR}};
    Conv2dConfig conv_config;
    conv_config.shard_layout = TensorMemoryLayout::BLOCK_SHARDED;
    conv_config.reshard_if_not_optimal = true;

    const auto input_plan = determine_conv_input_parallel_config(
        compute_grid,
        existing_memory_config,
        conv_config,
        /*batch_size=*/1,
        /*output_height=*/1,
        /*output_width=*/256,
        /*in_channels=*/64,
        /*out_channels=*/64,
        Layout::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*is_1d_depthwise_conv=*/false,
        /*l1_alignment_bytes=*/16,
        /*input_tensor_channels_padded=*/64);

    EXPECT_TRUE(input_plan.needs_shard_or_reshard);
    EXPECT_EQ(input_plan.input_channels_padded, 64);
    EXPECT_EQ(input_plan.parallel_config.shard_scheme, TensorMemoryLayout::BLOCK_SHARDED);
    EXPECT_EQ(input_plan.parallel_config.shard_orientation, ShardOrientation::ROW_MAJOR);
    EXPECT_EQ(input_plan.parallel_config.grid.bounding_box().grid_size(), (CoreCoord{8, 8}));
}

TEST(Conv2DInputPlannerTest, ReshardPreservesPhysicalChannelPadding) {
    const CoreCoord compute_grid{13, 10};

    const auto existing_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}};
    const auto existing_memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{existing_grid, {32, 40}, ShardOrientation::ROW_MAJOR}};
    Conv2dConfig conv_config;
    conv_config.shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    conv_config.override_sharding_config = true;
    conv_config.core_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 0}}};

    const auto input_plan = determine_conv_input_parallel_config(
        compute_grid,
        existing_memory_config,
        conv_config,
        /*batch_size=*/1,
        /*output_height=*/1,
        /*output_width=*/30,
        /*in_channels=*/24,
        /*out_channels=*/32,
        Layout::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*is_1d_depthwise_conv=*/false,
        /*l1_alignment_bytes=*/16,
        /*input_tensor_channels_padded=*/40);

    EXPECT_TRUE(input_plan.needs_shard_or_reshard);
    EXPECT_EQ(input_plan.input_channels_padded, 40);
    EXPECT_EQ(input_plan.parallel_config.grid, conv_config.core_grid.value());
}

TEST(Conv2DInputPlannerTest, HeightReshardPadsChannelsForTileLayout) {
    const CoreCoord compute_grid{13, 10};

    const auto existing_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}};
    const auto existing_memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{existing_grid, {1, 16}, ShardOrientation::ROW_MAJOR}};
    Conv2dConfig conv_config;
    conv_config.shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;

    const auto input_plan = determine_conv_input_parallel_config(
        compute_grid,
        existing_memory_config,
        conv_config,
        /*batch_size=*/3,
        /*output_height=*/1,
        /*output_width=*/1,
        /*in_channels=*/16,
        /*out_channels=*/32,
        Layout::ROW_MAJOR,
        /*is_mm_conv=*/true,
        /*is_1d_depthwise_conv=*/false,
        /*l1_alignment_bytes=*/64,
        /*input_tensor_channels_padded=*/16);

    EXPECT_TRUE(input_plan.needs_shard_or_reshard);
    EXPECT_EQ(input_plan.input_channels_padded, 32);
    EXPECT_EQ(input_plan.parallel_config.shard_scheme, TensorMemoryLayout::HEIGHT_SHARDED);
}

TEST(Conv2DInputPlannerTest, DepthwiseBlockReshardPreservesShardWidthAndTrimsInactiveCores) {
    const CoreCoord compute_grid{13, 10};

    const auto existing_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}};
    const auto existing_memory_config = MemoryConfig{
        TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{existing_grid, {32, 64}, ShardOrientation::ROW_MAJOR}};
    Conv2dConfig conv_config;
    conv_config.shard_layout = TensorMemoryLayout::BLOCK_SHARDED;

    const auto input_plan = determine_conv_input_parallel_config(
        compute_grid,
        existing_memory_config,
        conv_config,
        /*batch_size=*/1,
        /*output_height=*/1,
        /*output_width=*/29,
        /*in_channels=*/96,
        /*out_channels=*/96,
        Layout::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*is_1d_depthwise_conv=*/true,
        /*l1_alignment_bytes=*/16,
        /*input_tensor_channels_padded=*/128);

    EXPECT_TRUE(input_plan.needs_shard_or_reshard);
    EXPECT_EQ(input_plan.input_channels_padded, 128);
    EXPECT_EQ(input_plan.parallel_config.shard_scheme, TensorMemoryLayout::BLOCK_SHARDED);
    EXPECT_EQ(get_num_cores_channels_from_parallel_config(input_plan.parallel_config), 2);
}

TEST(Conv2DInputPlannerTest, DepthwiseBlockReshardUsesTileAlignedChannelPartition) {
    const CoreCoord compute_grid{13, 10};

    const auto existing_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}};
    const auto existing_memory_config = MemoryConfig{
        TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{existing_grid, {10, 16}, ShardOrientation::ROW_MAJOR}};
    Conv2dConfig conv_config;

    const auto input_plan = determine_conv_input_parallel_config(
        compute_grid,
        existing_memory_config,
        conv_config,
        /*batch_size=*/1,
        /*output_height=*/1,
        /*output_width=*/10,
        /*in_channels=*/64,
        /*out_channels=*/64,
        Layout::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*is_1d_depthwise_conv=*/true,
        /*l1_alignment_bytes=*/64,
        /*input_tensor_channels_padded=*/64);

    EXPECT_TRUE(input_plan.needs_shard_or_reshard);
    EXPECT_EQ(input_plan.input_channels_padded, 64);
    EXPECT_EQ(input_plan.parallel_config.shard_scheme, TensorMemoryLayout::BLOCK_SHARDED);
    EXPECT_EQ(get_num_cores_channels_from_parallel_config(input_plan.parallel_config), 2);
    EXPECT_EQ(
        input_plan.input_channels_padded / get_num_cores_channels_from_parallel_config(input_plan.parallel_config), 32);
}

TEST(Conv2DInputPlannerTest, DepthwiseBlockOutputPreservesInputChannelPartitionCapacity) {
    const CoreCoord compute_grid{13, 10};
    const auto source_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}};
    const auto source_memory_config = MemoryConfig{
        TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{source_grid, {32, 96}, ShardOrientation::ROW_MAJOR}};
    Conv2dConfig conv_config;
    conv_config.shard_layout = TensorMemoryLayout::BLOCK_SHARDED;

    const auto input_plan = determine_conv_input_parallel_config(
        compute_grid,
        source_memory_config,
        conv_config,
        /*batch_size=*/1,
        /*output_height=*/1,
        /*output_width=*/29,
        /*in_channels=*/100,
        /*out_channels=*/100,
        Layout::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*is_1d_depthwise_conv=*/true,
        /*l1_alignment_bytes=*/16,
        /*input_tensor_channels_padded=*/192);
    const auto output_plan = determine_output_parallel_config(
        input_plan.parallel_config,
        compute_grid,
        /*out_channels=*/100,
        ShardOrientation::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*require_input_channel_partition=*/true);

    EXPECT_EQ(get_num_cores_channels_from_parallel_config(input_plan.parallel_config), 2);
    EXPECT_EQ(input_plan.input_channels_padded, 192);
    EXPECT_EQ(
        determine_conv_output_channels_padded(
            input_plan.parallel_config,
            output_plan,
            input_plan.input_channels_padded,
            /*output_channels=*/100,
            /*is_1d_depthwise_conv=*/true),
        192);
}

TEST(Conv2DInputPlannerTest, DepthwiseBlockPreservesFullyPopulatedCustomWidthInput) {
    const CoreCoord compute_grid{13, 10};
    const auto source_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 0}}};
    const auto source_memory_config = MemoryConfig{
        TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{source_grid, {32, 96}, ShardOrientation::ROW_MAJOR}};
    Conv2dConfig conv_config;
    conv_config.shard_layout = TensorMemoryLayout::BLOCK_SHARDED;

    const auto input_plan = determine_conv_input_parallel_config(
        compute_grid,
        source_memory_config,
        conv_config,
        /*batch_size=*/1,
        /*output_height=*/1,
        /*output_width=*/29,
        /*in_channels=*/100,
        /*out_channels=*/100,
        Layout::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*is_1d_depthwise_conv=*/true,
        /*l1_alignment_bytes=*/16,
        /*input_tensor_channels_padded=*/192);

    EXPECT_FALSE(input_plan.needs_shard_or_reshard);
    EXPECT_EQ(input_plan.parallel_config.grid, source_grid);
    EXPECT_EQ(input_plan.input_channels_padded, 192);
}

TEST(Conv2DInputPlannerTest, CrossLayoutReshardUsesCanonicalChannelPadding) {
    const CoreCoord compute_grid{13, 10};

    const auto existing_grid = CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}};
    const auto existing_memory_config = MemoryConfig{
        TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec{existing_grid, {32, 32}, ShardOrientation::ROW_MAJOR}};
    Conv2dConfig conv_config;
    conv_config.shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    conv_config.reshard_if_not_optimal = true;

    const auto input_plan = determine_conv_input_parallel_config(
        compute_grid,
        existing_memory_config,
        conv_config,
        /*batch_size=*/1,
        /*output_height=*/1,
        /*output_width=*/29,
        /*in_channels=*/96,
        /*out_channels=*/96,
        Layout::ROW_MAJOR,
        /*is_mm_conv=*/false,
        /*is_1d_depthwise_conv=*/false,
        /*l1_alignment_bytes=*/16,
        /*input_tensor_channels_padded=*/96);

    EXPECT_TRUE(input_plan.needs_shard_or_reshard);
    EXPECT_EQ(input_plan.input_channels_padded, 96);
    EXPECT_EQ(input_plan.parallel_config.shard_scheme, TensorMemoryLayout::HEIGHT_SHARDED);
}

/*
    Reference implementation of Conv2D

    Takes in input tensor with original shape (N,Ci,H,W) that is flattened in row major order

    and flattened kernel tensor with original shape (Co,Ci,KH,KW) that is also flattened in row major order.

    Returns flattened tensor with original shape (N,Co,Xh,Xw) in row major order, where Xh and Xw are calculated based
    on input tensor,kernel tensor, stride and padding.


    The output vector is flattened in row major order.

    input_channels - Ci
    output_channels - Co
    input_height - H
    input_width - W
    batch_size - N
    output_height - Xh
    output_width - Xw
    kernel_size - (KH,KW)
    stride - (SH,SW)
    padding - (PH,PW)
*/
std::vector<float> reference_implementation_conv2d(
    const std::vector<float>& input,   // (N,Ci,H,W)
    const std::vector<float>& kernel,  // (Co,Ci,H',W')
    const uint32_t input_channels,
    const uint32_t output_channels,
    const uint32_t batch_size,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& padding) {
    uint32_t kernel_height = kernel_size[0];
    uint32_t kernel_width = kernel_size[1];
    uint32_t padding_height = padding[0];
    uint32_t padding_width = padding[1];
    uint32_t stride_height = stride[0];
    uint32_t stride_width = stride[1];

    // Calculate output height and width
    uint32_t Xh = ((input_height - kernel_height + 2 * padding_height) / stride_height) + 1;
    uint32_t Xw = ((input_width - kernel_width + 2 * padding_width) / stride_width) + 1;

    std::vector<float> output = std::vector<float>(batch_size * output_channels * Xh * Xw);
    uint32_t i = 0;
    for (uint32_t n = 0; n < batch_size; n++) {
        for (uint32_t co = 0; co < output_channels; co++) {
            for (uint32_t h = 0; h < input_height; h += stride_height) {
                std::vector<float> row;
                for (uint32_t w = 0; w < input_width; w += stride_width) {
                    float sum = 0;
                    for (uint32_t ci = 0; ci < input_channels; ci++) {
                        for (uint32_t kh = 0; kh < kernel_height; kh++) {
                            for (uint32_t kw = 0; kw < kernel_width; kw++) {
                                if (h + kh - padding_height >= 0 && h + kh - padding_height < input_height &&
                                    w + kw - padding_width >= 0 && w + kw - padding_width < input_width) {
                                    sum += input
                                               [(n * input_channels * input_height * input_width) +
                                                (ci * input_height * input_width) +
                                                ((h + kh - padding_height) * input_width) + w + kw - padding_width] *
                                           kernel
                                               [(co * input_channels * kernel_height * kernel_width) +
                                                (ci * kernel_height * kernel_width) + (kh * kernel_width) + kw];
                                }
                            }
                        }
                    }
                    output[i] = sum;
                    i++;
                }
            }
        }
    }
    return output;
}

TEST_P(Conv2DFixture, Conv2DCalculateCorrectly) {
    const Conv2DParam param = GetParam();

    // Sets the size for L1 small on the device - 16KB
    // The halo op which is contained in the Conv2D op uses L1 small memory
    // Without this, the convolution operation will fail due to L1_SMALL Out of Memory error
    const size_t l1_small_size = 16384;

    auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(
        /*device_id=*/0, l1_small_size);

    try {
        MemoryConfig dram_mem_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

        // (N,Ci,H,W)
        Shape dimensions{param.batch_size, param.input_channels, param.input_height, param.input_width};
        // (Co,Ci,KH,KW)
        Shape dimensions_weight{
            param.output_channels, param.input_channels, param.kernel_size[0], param.kernel_size[1]};

        random::seed(42);
        // Create input tensor on device
        Tensor input_tensor =
            ttnn::random::random(dimensions, tt::tt_metal::DataType::BFLOAT16).to_device(device.get(), dram_mem_config);

        // Create weight tensor on device (weight tensor on device would require to be tiled if
        // Conv2DConfig.always_preprocess_weights isn't used)
        Tensor weight_tensor = ttnn::random::random(dimensions_weight, tt::tt_metal::DataType::BFLOAT16);

        // Copy input tensor and weight tensor to host for reference implementation
        std::vector<float> input_vector = input_tensor.to_vector<float>();
        std::vector<float> weight_vector = weight_tensor.to_vector<float>();

        // (N,Ci,H,W) -> (N,H,W,Ci)
        input_tensor = ttnn::permute(input_tensor, ttsl::SmallVector<int64_t>{0, 2, 3, 1});

        // Run Conv2D
        auto [output_tensor, output_dimensions] = std::get<static_cast<int>(ConvResultType::OUTPUT_DIM)>(ttnn::conv2d(
            input_tensor,
            weight_tensor,
            device.get(),
            param.input_channels,
            param.output_channels,
            param.batch_size,
            param.input_height,
            param.input_width,
            param.kernel_size,
            param.stride,
            param.padding,
            std::array<uint32_t, 2>{1, 1},  // dilation
            1,                              // groups
            std::nullopt,                   // dtype
            std::nullopt,                   // bias tensor
            std::nullopt,                   // conv config
            std::nullopt,                   // compute config
            std::nullopt,                   // memory config
            std::nullopt,                   // slice config
            true                            // return_output_dim
            ));

        // move output tensor to dram
        output_tensor = ttnn::to_memory_config(output_tensor, dram_mem_config);

        // H'  - output_height
        // W'  - output_width
        // (1,1,NH'W',Co) -> (N,H',W',Co)
        output_tensor = ttnn::reshape(
            output_tensor,
            Shape(
                {param.batch_size,
                 std::get<0>(output_dimensions),
                 std::get<1>(output_dimensions),
                 param.output_channels}));

        // (N,H',W',Co) -> (N,Co,H',W')
        output_tensor = ttnn::permute(output_tensor, ttsl::SmallVector<int64_t>{0, 3, 1, 2});

        // Copy output tensor to host for comparison
        std::vector<float> res = output_tensor.to_vector<float>();

        // Run reference implementation of Conv2D
        std::vector<float> ref_res = reference_implementation_conv2d(
            input_vector,
            weight_vector,
            param.input_channels,
            param.output_channels,
            param.batch_size,
            param.input_height,
            param.input_width,
            param.kernel_size,
            param.stride,
            param.padding);

        EXPECT_GT(test_utils::pcc(res, ref_res), 0.99);
    } catch (const std::exception& e) {
        FAIL() << "Caught exception in Conv2D test: " << e.what();
        throw e;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Conv2DTests,
    Conv2DFixture,
    ::testing::Values(
        Conv2DParam{
            .input_channels = 3,
            .output_channels = 17,
            .batch_size = 5,
            .input_height = 111,
            .input_width = 25,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1},
        },
        Conv2DParam{
            .input_channels = 32,
            .output_channels = 32,
            .batch_size = 2,
            .input_height = 256,
            .input_width = 256,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1},
        },
        Conv2DParam{
            .input_channels = 3,
            .output_channels = 15,
            .batch_size = 7,
            .input_height = 3,
            .input_width = 3,
            .kernel_size = {3, 3},
            .stride = {1, 1},
            .padding = {1, 1},
        }));

}  // namespace ttnn::operations::conv::conv2d::test
