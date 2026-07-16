# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_prepare_conv_weights_matches_1x1_reshard_plan(device):
    if device.compute_with_storage_grid_size().x < 8 or device.compute_with_storage_grid_size().y < 4:
        pytest.skip("Test requires an 8x4 compute grid")

    torch.manual_seed(0)
    batch_size = 10
    input_height = 7
    input_width = 7
    input_channels = 960
    output_channels = 160
    torch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(output_channels, input_channels, 1, 1, dtype=torch.bfloat16).float()
    torch_bias = torch.randn(output_channels, dtype=torch.bfloat16).float()
    golden = torch.nn.functional.conv2d(torch_input, torch_weight, bias=torch_bias)

    input_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(input_grid, (128, 128), ttnn.ShardOrientation.ROW_MAJOR),
    )
    input_tt = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).reshape(1, 1, batch_size * input_height * input_width, input_channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias_tt = ttnn.from_torch(
        torch_bias.reshape(1, 1, 1, output_channels), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=True,
    )
    common_args = dict(
        input_memory_config=input_memory_config,
        input_layout=ttnn.TILE_LAYOUT,
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_config,
    )
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=weight_tt,
        weights_format="OIHW",
        has_bias=True,
        **common_args,
    )
    prepared_bias = ttnn.prepare_conv_bias(bias_tensor=bias_tt, **common_args)

    output_tt, [output_height, output_width] = ttnn.conv2d(
        input_tensor=input_tt,
        weight_tensor=prepared_weight,
        bias_tensor=prepared_bias,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(1, 1),
        conv_config=conv_config,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )
    assert output_tt.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    output = ttnn.to_torch(output_tt).reshape(batch_size, output_height, output_width, -1)
    output = output[..., :output_channels].permute(0, 3, 1, 2)
    passing, message = check_with_pcc_without_tensor_printout(output, golden, pcc=0.99)
    assert passing, message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_prepare_conv_weights_matches_auto_sharded_1x1_plan(device):
    torch.manual_seed(0)
    input_channels = 128
    output_channels = 64
    input_height = 8
    input_width = 8
    torch_input = torch.randn(1, input_channels, input_height, input_width, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(output_channels, input_channels, 1, 1, dtype=torch.bfloat16).float()
    golden = torch.nn.functional.conv2d(torch_input, torch_weight)

    input_tt = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
    common_args = dict(
        input_memory_config=ttnn.L1_MEMORY_CONFIG,
        input_layout=ttnn.TILE_LAYOUT,
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=1,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_config,
    )
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=weight_tt,
        weights_format="OIHW",
        has_bias=False,
        **common_args,
    )

    output_tt, [output_height, output_width] = ttnn.conv2d(
        input_tensor=input_tt,
        weight_tensor=prepared_weight,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        batch_size=1,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(1, 1),
        conv_config=conv_config,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )
    output = ttnn.to_torch(output_tt).reshape(1, output_height, output_width, -1)
    output = output[..., :output_channels].permute(0, 3, 1, 2)
    passing, message = check_with_pcc_without_tensor_printout(output, golden, pcc=0.999)
    assert passing, message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("override_sharding_config", [False, True], ids=["preserve", "reshard"])
def test_prepare_conv_weights_preserves_physical_channel_padding(device, override_sharding_config):
    torch.manual_seed(0)
    input_channels = 24
    output_channels = 32
    input_width = 32
    torch_input = torch.randn(1, input_channels, 1, input_width, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(output_channels, input_channels, 1, 3, dtype=torch.bfloat16).float()
    golden = torch.nn.functional.conv2d(torch_input, torch_weight)

    input_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(input_grid, (32, 40), ttnn.ShardOrientation.ROW_MAJOR),
    )
    input_tt = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    expected_grid = input_grid
    if override_sharding_config:
        expected_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
        conv_config.override_sharding_config = True
        conv_config.core_grid = expected_grid
    common_args = dict(
        input_memory_config=input_memory_config,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=1,
        input_height=1,
        input_width=input_width,
        kernel_size=(1, 3),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_config,
    )
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=weight_tt,
        weights_format="OIHW",
        has_bias=False,
        **common_args,
    )

    output_tt, [output_height, output_width] = ttnn.conv2d(
        input_tensor=input_tt,
        weight_tensor=prepared_weight,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        batch_size=1,
        input_height=1,
        input_width=input_width,
        kernel_size=(1, 3),
        conv_config=conv_config,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )
    assert output_tt.memory_config().shard_spec.grid == expected_grid
    output = ttnn.to_torch(output_tt).reshape(1, output_height, output_width, -1)
    output = output[..., :output_channels].permute(0, 3, 1, 2)
    passing, message = check_with_pcc_without_tensor_printout(output, golden, pcc=0.999)
    assert passing, message


def prepare_conv_weights_func(
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    config_override,
    device,
    groups,
    is_owned,
    slice_config=ttnn.Conv2dL1FullSliceConfig,
    weights_dtype=None,
    torch_weights_dtype=None,
    enable_kernel_stride_folding=False,
    input_layout=ttnn.ROW_MAJOR_LAYOUT,
    has_bias=False,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    if batch_size == 20 and (
        output_channels == 64 or (stride_h == 2 and (output_channels == 256 or output_channels == 128))
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    inp_shape = (batch_size, input_channels, input_height, input_width)
    conv_weight_shape = (output_channels, input_channels // groups, filter_height, filter_width)
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16)
    torch_input_tensor = torch.randn(inp_shape, dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn((1, 1, 1, output_channels), dtype=torch.bfloat16) if has_bias else None

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(1, 1),
        groups=groups,
    ).permute(0, 2, 3, 1)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.transpose(-3, -2).transpose(-2, -1),
        ttnn.bfloat16,
        layout=input_layout,
    )

    if is_owned:
        temp_tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
        temp_tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16) if has_bias else None
        tt_weight_tensor = ttnn.zeros(torch_weight_tensor.shape, ttnn.bfloat16)
        tt_bias_tensor = ttnn.zeros(torch_bias_tensor.shape, ttnn.bfloat16) if has_bias else None
        tt_weight_tensor = temp_tt_weight_tensor[:, :, :]
        tt_bias_tensor = temp_tt_bias_tensor[:, :, :] if has_bias else None
    else:
        tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
        tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16) if has_bias else None

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        enable_act_double_buffer=False,
        enable_kernel_stride_folding=enable_kernel_stride_folding,
    )
    compute_config = ttnn.init_device_compute_kernel_config(device.arch())
    if slice_config:
        compute_config.throttle_level = ttnn.ThrottleLevel(3)
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    conv_kwargs = {
        "input_layout": input_layout,
        "in_channels": input_channels,
        "out_channels": output_channels,
        "batch_size": batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "kernel_size": (filter_height, filter_width),
        "stride": (stride_h, stride_w),
        "padding": (pad_h, pad_w),
        "dilation": (1, 1),
        "groups": groups,
        "device": device,
        "conv_config": conv_config,
        "slice_config": slice_config,
    }

    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    tt_input_tensor = ttnn.to_device(tt_input_tensor, device)

    tt_weight_tensor_formatted = ttnn.prepare_conv_weights(
        weight_tensor=tt_weight_tensor,
        weights_format="OIHW",
        input_memory_config=input_memory_config,
        has_bias=has_bias,
        **conv_kwargs,
        input_dtype=ttnn.bfloat16,
    )
    tt_bias_tensor_formatted = (
        ttnn.prepare_conv_bias(
            bias_tensor=tt_bias_tensor,
            input_memory_config=input_memory_config,
            **conv_kwargs,
            input_dtype=ttnn.bfloat16,
        )
        if has_bias
        else None
    )
    tt_weight_tensor_formatted = ttnn.to_device(tt_weight_tensor_formatted, device)
    tt_bias_tensor_formatted = ttnn.to_device(tt_bias_tensor_formatted, device) if has_bias else None
    (k := next(iter(conv_kwargs)), conv_kwargs.pop(k))  ##removing 1st element from dict
    tt_output_tensor_on_device = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor_formatted,
        bias_tensor=tt_bias_tensor_formatted,
        **conv_kwargs,
        compute_config=compute_config,
        dtype=ttnn.bfloat16,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]
    torch_output_tensor = torch_output_tensor.reshape(torch_out_golden_tensor.shape)

    pcc = 0.99
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("num_slices", [1, 2], ids=["single_l1_slice", "dram_sharded_slices"])
def test_prepare_conv_weights_matches_dram_sliced_tile_plan(device, has_bias, num_slices):
    prepare_conv_weights_func(
        batch_size=1,
        output_channels=64,
        input_channels=24,
        input_height=64,
        input_width=64,
        filter_height=3,
        filter_width=3,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        config_override=None,
        device=device,
        groups=1,
        is_owned=False,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceWidth,
            num_slices=num_slices,
        ),
        input_layout=ttnn.TILE_LAYOUT,
        has_bias=has_bias,
        weights_dtype=ttnn.bfloat16,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("has_bias", [False, True])
def test_prepare_conv_weights_matches_dram_height_tile_width_padding_plan(device, has_bias):
    prepare_conv_weights_func(
        batch_size=1,
        output_channels=64,
        input_channels=24,
        input_height=64,
        input_width=50,
        filter_height=3,
        filter_width=3,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        config_override=None,
        device=device,
        groups=1,
        is_owned=False,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=ttnn.Conv2dDRAMSliceHeight,
            num_slices=2,
        ),
        input_layout=ttnn.TILE_LAYOUT,
        has_bias=has_bias,
        weights_dtype=ttnn.bfloat16,
    )


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w,  config_override, groups",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, None), HANGS!!
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, {"act_block_h": 256}, 1),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, {"act_block_h": 32}, 1),  Out of Memory!!
        # rn50 layer1
        (8, 64, 64, 56, 1, 3, 1, 1, 1, 1, 0, None, 64),
        (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None, 1),
        (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None, 1),
        # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, None, 1),
        (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, None, 1),
        (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, {"act_block_h": 32}, 1),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, None, 1),
        (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, None, 1),
        (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, None, 1),
        # rn50 layer3
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, None, 1),
        (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, None, 1),
        (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, None, 1),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, None, 1),
        (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, None, 1),
        (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, None, 1),
        # rn50 layer4
        (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, None, 1),
        (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, None, 1),
        (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, None, 1),
        (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, None, 1),
        (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, None, 1),
        (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, None, 1),
        ## small test
        (1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 2, "grid_size": (2, 2)}, 1),
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 4, "grid_size": (2, 4)}, 1),
        # (1, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, None, 1), sliding_window_op_infra/sliding_window.cpp:341: indices_length_last_core <= indices_length_per_core
        (8, 256, 256, 7, 7, 3, 3, 1, 1, 1, 1, None, 1),
        # r50 1x1s2 shapes
        # Fails with packer_l1_acc = True (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, None, 1),  # r50 first bottleneck downsample shape
        (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, None, 1),  # r50 first bottleneck downsample shape
        # Fails with packer_l1_acc = True (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, None, 1),  # r50 second bottleneck downsample shape
        # (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, True, None, 1), - doesnt fit
        (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, None, 1),  # r50 third bottleneck downsample shape
        # (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, True, None, 1), - doesnt fit
        (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, None, 1),  # r50 fourth bottleneck downsample shape
        # (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, True, None, 1), - doesnt fit
        # (20, 128, 256, 56, 56, 1, 1, 2, 2, 0, 0, True, None, 1),  ## L2M1 DS: doesn't fit
        # formerly failing test case in segformer when ntiles_channels not evenly divisible with num_cores_c
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, None, 1),
    ),
)
@pytest.mark.parametrize("is_owned", [True, False], ids=["owned_storage", "borrowed_storage"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2**15}], indirect=True)
def test_prepare_conv_weights(
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    config_override,
    device,
    groups,
    is_owned,
):
    prepare_conv_weights_func(
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        device,
        groups,
        is_owned,
    )


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w,  config_override, groups",
    (
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, {"act_block_h": 256}, 1),
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, None, 1),
    ),
)
@pytest.mark.parametrize("weights_dtype", [None, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("torch_weights_dtype", [ttnn.float32])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2**15}], indirect=True)
def test_conv_weights_dtype(
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    config_override,
    device,
    groups,
    weights_dtype,
    torch_weights_dtype,
):
    prepare_conv_weights_func(
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        device,
        groups,
        False,
        weights_dtype=weights_dtype,
        torch_weights_dtype=torch_weights_dtype,
    )


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override",
    (
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None),
        (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None),
        (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None),
        # formerly failing test case in segformer when ntiles_channels not evenly divisible with num_cores_c
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, None),
    ),
)
@pytest.mark.parametrize("has_bias", [True], ids=["has_bias"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2**15}], indirect=True)
def test_prepare_bias(
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    config_override,
    has_bias,
    device,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    if batch_size == 20 and (
        output_channels == 64 or (stride_h == 2 and (output_channels == 256 or output_channels == 128))
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    inp_shape = (batch_size, input_channels, input_height, input_width)
    conv_weight_shape = (output_channels, input_channels, filter_height, filter_width)
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16)
    torch_input_tensor = torch.randn(inp_shape, dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn((1, 1, 1, output_channels), dtype=torch.bfloat16) if has_bias else None

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(1, 1),
        groups=1,
    ).permute(0, 2, 3, 1)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor.transpose(-3, -2).transpose(-2, -1), ttnn.bfloat16)
    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
    tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16) if has_bias else None

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(device.arch())
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    conv_kwargs = {
        "input_layout": ttnn.ROW_MAJOR_LAYOUT,
        "in_channels": input_channels,
        "out_channels": output_channels,
        "batch_size": batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "kernel_size": (filter_height, filter_width),
        "stride": (stride_h, stride_w),
        "padding": (pad_h, pad_w),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    tt_input_tensor = ttnn.to_device(tt_input_tensor, device)

    tt_bias_tensor_formatted = (
        ttnn.prepare_conv_bias(
            bias_tensor=tt_bias_tensor,
            input_memory_config=ttnn.L1_MEMORY_CONFIG,
            **conv_kwargs,
            input_dtype=ttnn.bfloat16,
        )
        if has_bias
        else None
    )

    tt_bias_tensor_formatted = ttnn.to_device(tt_bias_tensor_formatted, device) if has_bias else None
    (k := next(iter(conv_kwargs)), conv_kwargs.pop(k))  ##removing 1st element from dict
    tt_output_tensor_on_device = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        bias_tensor=tt_bias_tensor_formatted,
        **conv_kwargs,
        compute_config=compute_config,
        dtype=ttnn.bfloat16,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]
    torch_output_tensor = torch_output_tensor.reshape(torch_out_golden_tensor.shape)

    pcc = 0.99
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


SliceHeight = ttnn.Conv2dDRAMSliceHeight
SliceWidth = ttnn.Conv2dDRAMSliceWidth


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, slice_type, num_slices, kernel, stride, padding, dilation, act_block_h_override",
    # fmt: off
    (
        (2, 64,   64,   384,   64,    SliceHeight,   6, (4, 4), (2, 2), (1, 1), (1, 1),  0,       ),
        (1, 32,   32,   1024,  1024,  SliceWidth,    4, (5, 5), (1, 1), (0, 0), (1, 1),  32,      ),
        (1, 64,   128,  992,   992,   SliceWidth,   31, (2, 2), (1, 1), (0, 0), (1, 1),  32 * 2,  ),
    )
    # fmt: on
)
def test_conv_dram(
    device,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    slice_type,
    num_slices,
    kernel,
    stride,
    padding,
    dilation,
    act_block_h_override,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }
    prepare_conv_weights_func(
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        config,
        device,
        groups=1,
        is_owned=False,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        ),
    )


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w",
    (
        (1, 1024, 3, 224, 224, 16, 16, 16, 16),
        (1, 1024, 3, 224, 224, 32, 32, 32, 32),
        (1, 192, 3, 512, 672, 16, 16, 16, 16),
        (1, 192, 3, 512, 672, 32, 32, 32, 32),
        (1, 768, 3, 384, 512, 32, 32, 32, 32),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2**15}], indirect=True)
def test_prepare_conv_weights_with_fold(
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    device,
):
    pad_h = 0
    pad_w = 0
    groups = 1

    prepare_conv_weights_func(
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        None,
        device,
        groups,
        is_owned=False,
        enable_kernel_stride_folding=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("input_num_cores", [1, 2], ids=["compact", "overprovisioned"])
@pytest.mark.parametrize("has_bias", [False, True])
def test_prepare_conv_weights_matches_height_sharded_fold_plan(device, has_bias, input_num_cores):
    torch.manual_seed(0)
    batch_size = 1
    input_height = 8
    input_width = 8
    input_channels = 3
    output_channels = 32
    kernel_size = (2, 2)
    stride = (2, 2)

    torch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(output_channels, input_channels, *kernel_size, dtype=torch.bfloat16).float()
    torch_bias = torch.randn(output_channels, dtype=torch.bfloat16).float() if has_bias else None
    golden = torch.nn.functional.conv2d(torch_input, torch_weight, bias=torch_bias, stride=stride)

    input_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(input_num_cores - 1, 0))}
    )
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(input_grid, (64, 8), ttnn.ShardOrientation.ROW_MAJOR),
    )
    input_tt = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias_tt = (
        ttnn.from_torch(
            torch_bias.reshape(1, 1, 1, output_channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        if has_bias
        else None
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=True,
    )
    common_args = dict(
        input_memory_config=input_memory_config,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_config,
    )
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=weight_tt,
        weights_format="OIHW",
        has_bias=has_bias,
        **common_args,
    )
    prepared_bias = ttnn.prepare_conv_bias(bias_tensor=bias_tt, **common_args) if has_bias else None

    output_tt, [output_height, output_width] = ttnn.conv2d(
        input_tensor=input_tt,
        weight_tensor=prepared_weight,
        bias_tensor=prepared_bias,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        conv_config=conv_config,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )
    output = ttnn.to_torch(output_tt).reshape(batch_size, output_height, output_width, -1)
    output = output[..., :output_channels].permute(0, 3, 1, 2)
    passing, message = check_with_pcc_without_tensor_printout(output, golden, pcc=0.999)
    assert passing, message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("has_bias", [False, True])
def test_prepare_conv_weights_matches_batched_padded_fold_plan(device, has_bias):
    torch.manual_seed(0)
    batch_size = 3
    input_height = 1
    input_width = 1
    input_channels = 8
    output_channels = 32
    kernel_size = (2, 1)
    stride = (2, 1)
    padding = (0, 1, 0, 0)

    torch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(output_channels, input_channels, *kernel_size, dtype=torch.bfloat16).float()
    torch_bias = torch.randn(output_channels, dtype=torch.bfloat16).float() if has_bias else None
    golden = torch.nn.functional.conv2d(
        torch.nn.functional.pad(torch_input, (0, 0, 0, 1)), torch_weight, bias=torch_bias, stride=stride
    )

    input_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(input_grid, (1, 8), ttnn.ShardOrientation.ROW_MAJOR),
    )
    input_tt = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias_tt = (
        ttnn.from_torch(
            torch_bias.reshape(1, 1, 1, output_channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        if has_bias
        else None
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_kernel_stride_folding=True,
    )
    common_args = dict(
        input_memory_config=input_memory_config,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1),
        groups=1,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_config,
    )
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=weight_tt,
        weights_format="OIHW",
        has_bias=has_bias,
        **common_args,
    )
    prepared_bias = ttnn.prepare_conv_bias(bias_tensor=bias_tt, **common_args) if has_bias else None

    output_tt, [output_height, output_width] = ttnn.conv2d(
        input_tensor=input_tt,
        weight_tensor=prepared_weight,
        bias_tensor=prepared_bias,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        conv_config=conv_config,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )
    output = ttnn.to_torch(output_tt).reshape(batch_size, output_height, output_width, -1)
    output = output[..., :output_channels].permute(0, 3, 1, 2)
    passing, message = check_with_pcc_without_tensor_printout(output, golden, pcc=0.999)
    assert passing, message
