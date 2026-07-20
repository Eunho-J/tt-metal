// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "ttnn/operations/conv/conv_transpose2d/prepare_conv_transpose2d_weights.hpp"

#include <cstdint>
#include <optional>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/sliding_window/sliding_window.hpp>
#include <ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp>
#include "conv2d/conv2d_utils.hpp"
#include "conv2d/device/conv2d_device_operation_types.hpp"
#include "conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace ttnn::operations::sliding_window;

namespace ttnn::operations::conv::conv_transpose2d {

using ttnn::operations::conv::conv2d::detail::prepare_conv_bias;
using ttnn::operations::conv::conv2d::detail::prepare_conv_weights;
using ttnn::prim::Conv2dConfig;
using ttnn::prim::Conv2dSliceConfig;

namespace {

struct ConvTranspose2dDRAMPrepPlan {
    MemoryConfig input_memory_config;
    std::array<uint32_t, 2> input_tensor_hw;
    ConvTranspose2dDimensions dimensions;
    Conv2dConfig conv_config;
};

ConvTranspose2dDRAMPrepPlan determine_conv_transpose2d_dram_prep_plan(
    op_slicing::OpSliceAttr& slice_attr,
    const Conv2dSliceConfig& slice_config,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t output_height,
    uint32_t output_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    const Conv2dConfig& conv_config) {
    const uint32_t output_sliced_dim =
        slice_config.slice_type == Conv2dSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;
    op_slicing::validate_slice_config(output_sliced_dim, conv_config.output_layout, slice_config);

    const auto first_output_slice =
        op_slicing::get_output_slice_range(output_sliced_dim, conv_config.output_layout, slice_config, 0);
    std::array<uint32_t, 2> output_slice_start = {0, 0};
    std::array<uint32_t, 2> output_slice_end = {output_height, output_width};
    if (slice_config.slice_type == Conv2dSliceConfig::SliceType::DRAM_HEIGHT) {
        output_slice_start[0] = first_output_slice.start;
        output_slice_end[0] = first_output_slice.end;
    } else {
        output_slice_start[1] = first_output_slice.start;
        output_slice_end[1] = first_output_slice.end;
    }

    const auto slice_plan = determine_conv_transpose2d_slice_plan(
        {input_height, input_width},
        output_slice_start,
        output_slice_end,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        conv_config.output_layout);
    auto input_memory_config = slice_attr.get_input_memory_config(
        {output_slice_start[0], output_slice_start[1]}, {output_slice_end[0], output_slice_end[1]});
    const uint32_t input_slice_height = slice_plan.input_end[0] - slice_plan.input_start[0];
    const uint32_t input_slice_width = slice_plan.input_end[1] - slice_plan.input_start[1];
    const auto dimensions = compute_conv_transpose2d_dimensions(
        input_slice_height,
        input_slice_width,
        kernel_size,
        stride,
        slice_plan.padding,
        slice_plan.output_padding,
        dilation);

    auto l1_conv_config = conv_config;
    if (!l1_conv_config.shard_layout.has_value()) {
        l1_conv_config.shard_layout = input_memory_config.memory_layout();
    }
    l1_conv_config.output_layout = Layout::TILE;
    return {
        .input_memory_config = std::move(input_memory_config),
        .input_tensor_hw = {input_slice_height, input_slice_width},
        .dimensions = dimensions,
        .conv_config = std::move(l1_conv_config)};
}

}  // namespace

// Compute all transposed conv2d dimension transformations in one place
// This uses SlidingWindowConfig as the single source of truth for how transposed conv2d
// parameters are transformed into conv2d parameters
ConvTranspose2dDimensions compute_conv_transpose2d_dimensions(
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation) {
    // Create SlidingWindowConfig with transposed conv parameters
    SlidingWindowConfig config;
    config.batch_size = 1;  // Batch size not needed for dimension calculations
    config.input_hw = {input_height, input_width};
    config.window_hw = {kernel_size[0], kernel_size[1]};
    config.stride_hw = {stride[0], stride[1]};
    config.padding = get_pair_n4_padding(padding);
    config.output_pad_hw = {output_padding[0], output_padding[1]};
    config.dilation_hw = {dilation[0], dilation[1]};
    config.is_transpose = true;

    // Use SlidingWindowConfig methods to compute dimensions
    auto output_shape = config.get_output_shape();
    auto full_input_shape = config.get_transposed_full_input_shape();
    auto real_padding = config.get_transposed_real_padding();

    // Calculate strided dimensions (not exposed by SlidingWindowConfig, but simple formula)
    uint32_t strided_input_height = ((input_height - 1) * stride[0]) + 1;
    uint32_t strided_input_width = ((input_width - 1) * stride[1]) + 1;

    // Populate result struct
    ConvTranspose2dDimensions dims{};
    dims.output_height = output_shape[1];
    dims.output_width = output_shape[2];
    dims.full_input_height = full_input_shape[1];
    dims.full_input_width = full_input_shape[2];
    dims.strided_input_height = strided_input_height;
    dims.strided_input_width = strided_input_width;
    dims.input_pad_top = real_padding[0].first;
    dims.input_pad_bottom = real_padding[0].second;
    dims.input_pad_left = real_padding[1].first;
    dims.input_pad_right = real_padding[1].second;

    return dims;
}

template <typename T>
ttnn::Tensor _transform_weights_for_conv_transpose2d(const Tensor& conv_weight_tensor, bool mirror_kernel = true) {
    TT_FATAL(is_cpu_tensor(conv_weight_tensor), "transform_weights_for_conv_transpose2d only supports host tensors");

    // in_w_shape = {in_channels, out_channels, kernel_height, kernel_width}
    // out_w_shape = {out_channels, in_channels, kernel_height, kernel_width}
    // Flip kernel_height and kernel_width
    const auto& in_w_shape = conv_weight_tensor.padded_shape();
    const uint32_t in_channels = in_w_shape[0];
    const uint32_t out_channels = in_w_shape[1];
    const uint32_t kernel_height = in_w_shape[2];
    const uint32_t kernel_width = in_w_shape[3];
    const ttnn::Shape output_shape{out_channels, in_channels, kernel_height, kernel_width};
    auto compute = [&output_shape, in_channels, out_channels, kernel_height, kernel_width, mirror_kernel](
                       const tt::tt_metal::HostBuffer& input_host_buffer) {
        auto input_buffer = tt::tt_metal::host_buffer::get_as<T>(input_host_buffer);
        auto owned_buffer = std::vector<T>(output_shape.volume());

        for (uint32_t out_channels_index = 0; out_channels_index < out_channels; out_channels_index++) {
            uint32_t output_weight_out_channel_base_idx =
                out_channels_index * in_channels * kernel_height * kernel_width;
            uint32_t input_weight_out_channel_base_idx = out_channels_index * kernel_height * kernel_width;
            for (uint32_t in_channels_index = 0; in_channels_index < in_channels; in_channels_index++) {
                uint32_t output_weight_in_channel_base_idx = in_channels_index * kernel_height * kernel_width;
                uint32_t input_weight_in_channel_base_idx =
                    in_channels_index * kernel_height * kernel_width * out_channels;

                for (uint32_t in_kernel_height_index = 0; in_kernel_height_index < kernel_height;
                     in_kernel_height_index++) {
                    uint32_t out_buffer_kh_index =
                        mirror_kernel ? kernel_height - in_kernel_height_index - 1 : in_kernel_height_index;
                    uint32_t in_height_offset = in_kernel_height_index * kernel_width;
                    uint32_t out_height_offset = out_buffer_kh_index * kernel_width;
                    for (uint32_t in_kernel_width_index = 0; in_kernel_width_index < kernel_width;
                         in_kernel_width_index++) {
                        uint32_t out_buffer_kw_index =
                            mirror_kernel ? kernel_width - in_kernel_width_index - 1 : in_kernel_width_index;

                        uint32_t in_idx = input_weight_out_channel_base_idx + input_weight_in_channel_base_idx +
                                          in_height_offset + in_kernel_width_index;
                        uint32_t out_idx = output_weight_out_channel_base_idx + output_weight_in_channel_base_idx +
                                           out_height_offset + out_buffer_kw_index;

                        owned_buffer[out_idx] = input_buffer[in_idx];
                    }
                }
            }
        }
        return tt::tt_metal::HostBuffer(std::move(owned_buffer));
    };

    const TensorSpec output_spec(
        output_shape,
        tt::tt_metal::TensorLayout(
            conv_weight_tensor.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    auto transformed_buffer = conv_weight_tensor.host_storage().buffer().transform(
        compute, tt::tt_metal::DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);
    return Tensor(tt::tt_metal::HostTensor::from_buffer(
        std::move(transformed_buffer), output_spec, conv_weight_tensor.tensor_topology()));
}

Tensor transform_weights_for_conv_transpose2d(const Tensor& conv_weight_tensor, bool mirror_kernel) {
    Tensor to_mirror_tensor;
    if (tt::tt_metal::is_device_tensor(conv_weight_tensor)) {
        log_warning(
            tt::LogOp,
            "Prepare Weights for ConvTranspose2D needs weights on host, but they are already on device. The op will "
            "move them back to host.");
        to_mirror_tensor = ttnn::operations::core::from_device(conv_weight_tensor);
    } else {
        to_mirror_tensor = conv_weight_tensor;
    }
    switch (conv_weight_tensor.dtype()) {
        case DataType::BFLOAT16:
            return _transform_weights_for_conv_transpose2d<::bfloat16>(to_mirror_tensor, mirror_kernel);
        case DataType::FLOAT32: return _transform_weights_for_conv_transpose2d<float>(to_mirror_tensor, mirror_kernel);
        case DataType::UINT32:
            return _transform_weights_for_conv_transpose2d<uint32_t>(to_mirror_tensor, mirror_kernel);
        default: TT_THROW("Unsupported data type for transform_weights_for_conv_transpose2d", to_mirror_tensor.dtype());
    }
};

ttnn::Tensor prepare_conv_transpose2d_weights(
    const ttnn::Tensor& weight_tensor,
    ttnn::MemoryConfig input_memory_config,
    Layout input_layout,
    const std::string& weights_format,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    const bool has_bias,
    uint32_t groups,
    MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_,
    bool mirror_kernel) {
    auto padding_n4 = sliding_window::get_pair_n4_padding(padding);
    DataType conv_output_dtype = output_dtype.value_or(input_dtype);
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const bool allow_matmul = use_matmul_for_1x1_conv(
        kernel_size,
        stride,
        padding_n4,
        dilation,
        groups,
        conv_config,
        input_memory_config.is_sharded() ? std::make_optional(input_memory_config) : std::nullopt);
    // Use weights_dtype from config if set, otherwise use weight tensor's dtype
    DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype());
    conv_config.weights_dtype = weight_dtype;
    DeviceComputeKernelConfig compute_config =
        compute_config_.value_or(get_conv_default_compute_kernel_config(device, input_dtype, weight_dtype));
    TT_ASSERT(
        weights_format == "IOHW",
        "PyTorch expects weights for ConvTranspose2D in IOHW format. If you have passed the correct weights, then make "
        "sure that the weights_format string is set to \"IOHW\".");

    // For grouped conv_transpose2d (groups > 1), we need to:
    // 1. Apply grouped layout conversion BEFORE the transpose to expand the weight tensor
    // 2. Then apply the standard transpose transformation
    // 3. Use groups=1 for the rest of the pipeline since grouping is already handled
    Tensor weight_for_transform = weight_tensor;
    uint32_t groups_for_prep = groups;
    if (groups > 1) {
        // Convert [in_channels, out_channels/groups, H, W] -> [in_channels, out_channels, H, W]
        weight_for_transform = conv2d::convert_conv_weight_tensor_to_grouped_layout_for_conv_transpose2d(
            weight_tensor, groups, weight_tensor.dtype());
        // After grouped conversion, we use groups=1 since the grouping is already embedded in the weights
        groups_for_prep = 1;
    }

    // Non-matmul auto-slicing must be resolved before selecting the execution path.
    // Matmul follows the runtime's direct L1 planning path and does not consume a slice plan.
    std::optional<Conv2dSliceConfig> actual_slice_config = dram_slice_config_;
    if (!allow_matmul && dram_slice_config_.has_value() && dram_slice_config_.value().num_slices == 0) {
        // Need to auto-determine - create temporary structures.
        // output_padding must match the value the conv op will use so that the auto-determined
        // slice count agrees between weight preparation and the actual conv. A mismatch (e.g.
        // assuming 0 here while the op uses a non-zero output_padding) changes the output image
        // size, which can tip the auto-slicer to a different num_slices and lay the weights out
        // for the wrong blocking, producing near-zero PCC.
        auto [output_height, output_width] = calculate_ct2d_output_image_size(
            {input_height, input_width}, kernel_size, stride, padding_n4, output_padding, dilation);

        auto temp_slice_attr = get_conv_transpose2d_slice_attr(
            batch_size,
            input_height,
            input_width,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding_n4,
            output_padding,
            dilation,
            groups,
            input_layout,
            input_dtype,
            conv_output_dtype,
            weight_dtype,
            kernel_size[1],
            has_bias,
            conv_config,
            compute_config,
            device,
            mirror_kernel);

        actual_slice_config = op_slicing::determine_slice_config(
            temp_slice_attr.get(),
            ttnn::Shape{batch_size, input_height, input_width, in_channels},
            ttnn::Shape{batch_size, output_height, output_width, out_channels},
            dram_slice_config_,
            conv_config.output_layout,
            device);

        // If auto-determination resulted in num_slices==1, convert to L1_FULL to avoid DRAM overhead
        if (actual_slice_config.has_value() && actual_slice_config.value().num_slices == 1) {
            actual_slice_config =
                op_slicing::Op2DSliceConfig{.slice_type = op_slicing::Op2DSliceConfig::SliceType::L1_FULL};
            log_debug(
                tt::LogOp,
                "Auto determined num_slices=1, converting to L1_FULL for ConvTranspose2d Weights {}",
                temp_slice_attr->name());
        } else {
            log_debug(
                tt::LogOp,
                "Auto determined DRAM Slice Config in Prepare Conv_Transpose2d Weights as {} for {}",
                actual_slice_config.value(),
                temp_slice_attr->name());
        }
    }

    // Determine execution path based on configuration and input properties
    ConvT2dExecutionPath path = allow_matmul
                                    ? ConvT2dExecutionPath::L1
                                    : determine_conv_transpose2d_execution_path(
                                          tt::tt_metal::StorageType::DEVICE, input_memory_config, actual_slice_config);

    Tensor mirrored_weight_tensor = transform_weights_for_conv_transpose2d(weight_for_transform, mirror_kernel);
    if (path == ConvT2dExecutionPath::L1) {
        // For transposed conv2d, the conv2d micro-op always uses stride=1x1 and operates on
        // "full_input" dimensions (after halo/padding expansion), not the original input dimensions.
        auto dims = compute_conv_transpose2d_dimensions(
            input_height, input_width, kernel_size, stride, padding, output_padding, dilation);

        return prepare_conv_weights(
            mirrored_weight_tensor,
            input_memory_config,
            input_layout,
            "OIHW",  // transform_weights_for_conv_transpose2d already converted IOHW -> OIHW
            in_channels,
            out_channels,
            batch_size,
            dims.full_input_height,  // Use full_input dimensions, not original
            dims.full_input_width,   // Use full_input dimensions, not original
            kernel_size,
            ConvTranspose2dDimensions::CONV2D_STRIDE,   // stride is always 1x1 for conv2d micro-op
            ConvTranspose2dDimensions::CONV2D_PADDING,  // padding is 0 (halo already added padding)
            dilation,
            has_bias,
            groups_for_prep,  // Use 1 if groups > 1 since grouped conversion is already done
            device,
            input_dtype,
            output_dtype,
            conv_config_,
            compute_config_,
            op_slicing::Op2DSliceConfig{.slice_type = op_slicing::Op2DSliceConfig::SliceType::L1_FULL},
            allow_matmul,
            false,
            std::array<uint32_t, 2>{input_height, input_width});
    }

    // DRAM path continues - need to set up slice configuration
    auto [output_height, output_width] = calculate_ct2d_output_image_size(
        {input_height, input_width}, kernel_size, stride, padding_n4, output_padding, dilation);
    auto convt2d_slice_attr = get_conv_transpose2d_slice_attr(
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding_n4,
        output_padding,
        dilation,
        groups,
        input_layout,
        input_dtype,
        conv_output_dtype,
        weight_dtype,
        kernel_size[1],
        has_bias,
        conv_config,
        compute_config,
        device,
        mirror_kernel);

    // Use the actual_slice_config if we determined it earlier, otherwise determine it now
    Conv2dSliceConfig dram_slice_config;
    if (actual_slice_config.has_value()) {
        dram_slice_config = actual_slice_config.value();
    } else {
        dram_slice_config = op_slicing::determine_slice_config(
            convt2d_slice_attr.get(),
            ttnn::Shape{batch_size, input_height, input_width, in_channels},
            ttnn::Shape{batch_size, output_height, output_width, out_channels},
            dram_slice_config_,
            conv_config.output_layout,
            device);
        log_debug(
            tt::LogOp,
            "DRAM Slice Config in Prepare Conv_Transpose2d Weights: {} for {}",
            dram_slice_config,
            convt2d_slice_attr->name());
    }

    const auto prep_plan = determine_conv_transpose2d_dram_prep_plan(
        *convt2d_slice_attr,
        dram_slice_config,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding_n4,
        output_padding,
        dilation,
        conv_config);

    return prepare_conv_weights(
        mirrored_weight_tensor,
        prep_plan.input_memory_config,
        Layout::ROW_MAJOR,
        "OIHW",  // transform_weights_for_conv_transpose2d already converted IOHW -> OIHW
        in_channels,
        out_channels,
        batch_size,
        prep_plan.dimensions.full_input_height,
        prep_plan.dimensions.full_input_width,
        kernel_size,
        ConvTranspose2dDimensions::CONV2D_STRIDE,   // stride is always 1x1 for conv2d micro-op
        ConvTranspose2dDimensions::CONV2D_PADDING,  // padding is 0 (halo already added padding)
        dilation,
        has_bias,
        groups_for_prep,  // Use 1 if groups > 1 since grouped conversion is already done
        device,
        input_dtype,
        output_dtype,
        prep_plan.conv_config,
        compute_config_,
        op_slicing::Op2DSliceConfig{.slice_type = op_slicing::Op2DSliceConfig::SliceType::L1_FULL},
        allow_matmul,
        false,
        prep_plan.input_tensor_hw);
}

ttnn::Tensor prepare_conv_transpose2d_bias(
    const ttnn::Tensor& bias_tensor,
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    MeshDevice* device,
    DataType input_dtype,
    const std::optional<const DataType>& output_dtype,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_) {
    const uint32_t groups_for_prep = groups > 1 ? 1 : groups;
    const auto padding_n4 = sliding_window::get_pair_n4_padding(padding);
    const Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType conv_output_dtype = output_dtype.value_or(input_dtype);
    const DataType weight_dtype = conv2d::detail::get_conv_bias_weight_dtype(conv_config);
    const DeviceComputeKernelConfig compute_config =
        compute_config_.value_or(get_conv_default_compute_kernel_config(device, input_dtype, weight_dtype));
    const bool allow_matmul = use_matmul_for_1x1_conv(
        kernel_size,
        stride,
        padding_n4,
        dilation,
        groups,
        conv_config,
        input_memory_config.is_sharded() ? std::make_optional(input_memory_config) : std::nullopt);
    std::optional<Conv2dSliceConfig> actual_slice_config = dram_slice_config_;
    ConvT2dExecutionPath path = allow_matmul
                                    ? ConvT2dExecutionPath::L1
                                    : determine_conv_transpose2d_execution_path(
                                          tt::tt_metal::StorageType::DEVICE, input_memory_config, actual_slice_config);

    std::unique_ptr<op_slicing::OpSliceAttr> slice_attr;
    uint32_t output_height = 0;
    uint32_t output_width = 0;
    if (path == ConvT2dExecutionPath::DRAM) {
        std::tie(output_height, output_width) = calculate_ct2d_output_image_size(
            {input_height, input_width}, kernel_size, stride, padding_n4, output_padding, dilation);
        slice_attr = get_conv_transpose2d_slice_attr(
            batch_size,
            input_height,
            input_width,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding_n4,
            output_padding,
            dilation,
            groups,
            input_layout,
            input_dtype,
            conv_output_dtype,
            weight_dtype,
            kernel_size[1],
            true,
            conv_config,
            compute_config,
            device,
            true);
        actual_slice_config = op_slicing::determine_slice_config(
            slice_attr.get(),
            ttnn::Shape{batch_size, input_height, input_width, in_channels},
            ttnn::Shape{batch_size, output_height, output_width, out_channels},
            dram_slice_config_,
            conv_config.output_layout,
            device);
        if (actual_slice_config->num_slices == 1) {
            path = ConvT2dExecutionPath::L1;
        }
    }

    MemoryConfig prep_input_memory_config = input_memory_config;
    Layout prep_input_layout = input_layout;
    auto prep_conv_config = conv_config;
    std::array<uint32_t, 2> prep_input_tensor_hw = {input_height, input_width};
    auto dims = compute_conv_transpose2d_dimensions(
        input_height, input_width, kernel_size, stride, padding, output_padding, dilation);
    if (path == ConvT2dExecutionPath::DRAM) {
        const auto prep_plan = determine_conv_transpose2d_dram_prep_plan(
            *slice_attr,
            *actual_slice_config,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding_n4,
            output_padding,
            dilation,
            conv_config);
        prep_input_memory_config = prep_plan.input_memory_config;
        prep_input_layout = Layout::ROW_MAJOR;
        prep_conv_config = prep_plan.conv_config;
        prep_input_tensor_hw = prep_plan.input_tensor_hw;
        dims = prep_plan.dimensions;
    }

    return prepare_conv_bias(
        bias_tensor,
        prep_input_memory_config,
        prep_input_layout,
        in_channels,
        out_channels,
        batch_size,
        dims.full_input_height,
        dims.full_input_width,
        kernel_size,
        ConvTranspose2dDimensions::CONV2D_STRIDE,   // stride is always 1x1 for conv2d micro-op
        ConvTranspose2dDimensions::CONV2D_PADDING,  // padding is 0 (halo already added padding)
        dilation,
        groups_for_prep,
        device,
        input_dtype,
        output_dtype,
        prep_conv_config,
        compute_config_,
        op_slicing::Op2DSliceConfig{.slice_type = op_slicing::Op2DSliceConfig::SliceType::L1_FULL},
        allow_matmul,
        prep_input_tensor_hw);
}

}  // namespace ttnn::operations::conv::conv_transpose2d
