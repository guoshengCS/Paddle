/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <chrono>  // NOLINT
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedDropoutDescriptor = platform::ScopedDropoutDescriptor;
using DataLayout = platform::DataLayout;
template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;

// Timer for timer
class Timer {
 public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

template <typename T>
class DropoutCUDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");

    const Tensor *input = ctx.Input<Tensor>("X");
    // Tensor *mask = ctx.Output<Tensor>("Mask");
    Tensor *output = ctx.Output<Tensor>("Out");

    const T *input_data = input->data<T>();
    // T *mask_data = output->mutable_data<T>(ctx.GetPlace());
    T *output_data = output->mutable_data<T>(ctx.GetPlace());

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_test = ctx.Attr<bool>("is_test");

    if (is_test) {
      return;
    }

    auto size_prod = input->numel();

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PushEvent("cudnnDropoutPrepare", &dev_ctx);
    auto handle = dev_ctx.cudnn_handle();
    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor data_desc;
    cudnnTensorDescriptor_t cudnn_data_desc =
        data_desc.descriptor<T>(GetCudnnTensorFormat(DataLayout::kNCHW),
                                paddle::platform::CudnnDataType<T>::type,
                                size_prod, 1, 1, 1);
    size_t reserve_space_size_in_bytes;
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetReserveSpaceSize(
        cudnn_data_desc, &reserve_space_size_in_bytes));
    Tensor unit8_mask;
    unit8_mask.Resize({static_cast<int64_t>(reserve_space_size_in_bytes)});
    uint8_t *unit8_mask_data = unit8_mask.mutable_data<uint8_t>(ctx.GetPlace());
    size_t states_size_in_bytes;
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetStatesSize(
        handle, &states_size_in_bytes));
    Tensor states;
    states.Resize({static_cast<int64_t>(states_size_in_bytes)});
    uint8_t *states_data = states.mutable_data<uint8_t>(ctx.GetPlace());
    std::random_device rnd;
    int seed = ctx.Attr<bool>("fix_seed") ? ctx.Attr<int>("seed") : rnd();
    // Timer timer;
    // timer.tic();
    ScopedDropoutDescriptor dropout_desc;
    cudnnDropoutDescriptor_t cudnn_dropout_desc = dropout_desc.descriptor(
        handle, dropout_prob, states_data, states_size_in_bytes,
        static_cast<uint64_t>(seed));
    PopEvent("cudnnDropoutPrepare", &dev_ctx);
    // std::cout << timer.toc();
    PushEvent("cudnnDropoutForward", &dev_ctx);
    // ------------------- cudnn dropout algorithm ---------------------
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutForward(
        handle, cudnn_dropout_desc, cudnn_data_desc, input_data,
        cudnn_data_desc, output_data, unit8_mask_data,
        reserve_space_size_in_bytes));
    PopEvent("cudnnDropoutForward", &dev_ctx);
  }

 protected:
  cudnnTensorDescriptor_t data_desc_;
  cudnnDropoutDescriptor_t dropout_desc_;
};

template <typename T>
class DropoutCUDNNGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_KERNEL(dropout, CUDNN, plat::CUDAPlace,
                   ops::DropoutCUDNNOpKernel<float>);
REGISTER_OP_KERNEL(dropout_grad, CUDNN, plat::CUDAPlace,
                   ops::DropoutCUDNNGradOpKernel<float>);
