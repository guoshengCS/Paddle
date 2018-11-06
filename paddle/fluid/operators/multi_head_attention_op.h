/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <random>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/platform/for_range.h"

#ifdef PADDLE_WITH_CUDA
#include <thrust/random.h>
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T, typename D>
struct DropoutFunctor {
  inline DropoutFunctor(T *in, T *out, D *mask, float dropout_prob, int seed);

  HOSTDEVICE inline T operator()(int64_t idx);

  T *in_;
  T *out_;
  D *mask_;
  float dropout_prob_;
  int seed_;
};

template <typename T, typename D>
struct DropoutFunctor<platform::CPUDeviceContext, T, D> {
  inline DropoutFunctor(T *in, T *out, D *mask, float dropout_prob, int seed)
      : in_(in),
        out_(out),
        mask_(mask),
        dropout_prob_(dropout_prob),
        seed_(seed) {
    engine_.seed(seed);
  }

  HOSTDEVICE inline T operator()(int64_t idx) {
    if (dist_(engine_) < dropout_prob_) {
      mask_[idx] = static_cast<D>(0);
      out_[idx] = 0;
    } else {
      mask_[idx] = static_cast<D>(1);
      out_[idx] = in_[idx];
    }
    return out_[idx];
  }

  T *in_;
  T *out_;
  D *mask_;
  float dropout_prob_;
  int seed_;
  std::minstd_rand engine_;
  std::uniform_real_distribution<float> dist_;  // (0, 1);
};

#ifdef PADDLE_WITH_CUDA
template <typename T, typename D>
struct DropoutFunctor<platform::CUDADeviceContext, T, D> {
  inline DropoutFunctor(T *in, T *out, D *mask, float dropout_prob, int seed)
      : in_(in),
        out_(out),
        mask_(mask),
        dropout_prob_(dropout_prob),
        seed_(seed) {}

  HOSTDEVICE inline T operator()(int64_t idx) {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    rng.discard(idx);
    if (dist_(rng) < dropout_prob_) {
      mask_[idx] = static_cast<D>(0);
      out_[idx] = 0;
    } else {
      mask_[idx] = static_cast<D>(1);
      out_[idx] = in_[idx];
    }
    return out_[idx];
  }

  T *in_;
  T *out_;
  D *mask_;
  float dropout_prob_;
  int seed_;
  thrust::uniform_real_distribution<float> dist_;  // (0, 1);
};
#endif

template <typename T>
struct ScaleFunctor {
  inline ScaleFunctor(T *in, T *out, T scale)
      : in_(in), out_(out), scale_(scale) {}

  HOSTDEVICE inline T operator()(int64_t idx) {
    return out_[idx] = in_[idx] * scale_;
  }

  T *in_;
  T *out_;
  T scale_;
};

template <typename DeviceContext, typename T>
class MultiHeadAttentionKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &dev_ctx = context.template device_context<DeviceContext>();

    auto *q = context.Input<framework::Tensor>("Q");
    auto *k = context.Input<framework::Tensor>("K");
    auto *v = context.Input<framework::Tensor>("V");
    auto *weight_q = context.Input<framework::Tensor>("Weight_Q");
    auto *weight_k = context.Input<framework::Tensor>("Weight_K");
    auto *weight_v = context.Input<framework::Tensor>("Weight_V");
    auto *proj_q = context.Output<framework::Tensor>("Proj_Q");
    auto *proj_k = context.Output<framework::Tensor>("Proj_K");
    auto *proj_v = context.Output<framework::Tensor>("Proj_V");
    auto *cache_k = context.Input<framework::Tensor>("Cache_K");
    auto *cache_v = context.Input<framework::Tensor>("Cache_V");
    auto *concat_k = context.Output<framework::Tensor>("Concat_K");
    auto *concat_v = context.Output<framework::Tensor>("Concat_V");
    auto *dot_product = context.Output<framework::Tensor>("Dot_Product");
    auto *attn_bias = context.Input<framework::Tensor>("Attn_Bias");
    auto *attn_weight = context.Output<framework::Tensor>("Attn_Weight");
    auto *mask = context.Output<framework::Tensor>("Mask");
    auto *attn_context = context.Output<framework::Tensor>("Attn_Context");
    auto *weight_out = context.Input<framework::Tensor>("Weight_Out");
    auto *out = context.Output<framework::Tensor>("Out");
    proj_q->mutable_data<T>(context.GetPlace());
    proj_k->mutable_data<T>(context.GetPlace());
    proj_v->mutable_data<T>(context.GetPlace());
    dot_product->mutable_data<T>(context.GetPlace());
    mask->mutable_data<uint8_t>(context.GetPlace());
    attn_context->mutable_data<T>(context.GetPlace());
    out->mutable_data<T>(context.GetPlace());

    auto scale = static_cast<T>(context.Attr<float>("scale"));
    auto n_head = context.Attr<int>("n_head");
    auto batch_size = q->dims()[0];
    auto src_seq_len = q->dims()[1];
    auto trg_seq_len = k->dims()[1];
    // auto d_model = q->dims()[2];
    auto d_key = weight_k->dims()[1] / n_head;
    auto d_value = weight_v->dims()[1] / n_head;

    // compute_qkv
    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_q = math::CreateMatrixDescriptor(
        q->dims(), /* num_flatten_cols */ 2, /* trans */ false);
    auto mat_dim_weight_q = math::CreateMatrixDescriptor(
        weight_q->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    blas.MatMul(*q, mat_dim_q, *weight_q, mat_dim_weight_q, T(scale), proj_q,
                T(0));
    auto mat_dim_k = math::CreateMatrixDescriptor(
        k->dims(), /* num_flatten_cols */ 2, /* trans */ false);
    auto mat_dim_weight_k = math::CreateMatrixDescriptor(
        weight_k->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    blas.MatMul(*k, mat_dim_k, *weight_k, mat_dim_weight_k, T(1), proj_k, T(0));
    auto mat_dim_v = math::CreateMatrixDescriptor(
        v->dims(), /* num_flatten_cols */ 2, /* trans */ false);
    auto mat_dim_w_v = math::CreateMatrixDescriptor(
        weight_v->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    blas.MatMul(*v, mat_dim_v, *weight_v, mat_dim_w_v, T(1), proj_v, T(0));

    if (cache_k && cache_v) {
      // use cache and concat time steps
      trg_seq_len += cache_k->dims()[1];
      concat_k->mutable_data<T>(context.GetPlace());
      concat_v->mutable_data<T>(context.GetPlace());
      paddle::operators::math::ConcatFunctor<DeviceContext, T> concat_functor;
      concat_functor(dev_ctx, {*proj_k, *cache_k}, 1, concat_k);
      concat_functor(dev_ctx, {*proj_v, *cache_v}, 1, concat_v);
      proj_k = concat_k;
      proj_v = concat_v;
    }

    // split_heads
    proj_q->Resize({batch_size, src_seq_len, n_head, d_key});
    proj_k->Resize({batch_size, trg_seq_len, n_head, d_key});
    proj_v->Resize({batch_size, trg_seq_len, n_head, d_value});
    math::Transpose<DeviceContext, T, 4> trans;
    std::vector<int> axis{0, 2, 1, 3};
    trans(dev_ctx, *proj_q, proj_q, axis);
    trans(dev_ctx, *proj_k, proj_k, axis);
    trans(dev_ctx, *proj_v, proj_v, axis);
    proj_q->Resize({batch_size, n_head, src_seq_len, d_key});
    proj_k->Resize({batch_size, n_head, trg_seq_len, d_key});
    proj_v->Resize({batch_size, n_head, trg_seq_len, d_value});

    // dot_product_attention
    auto mat_dim_proj_q = math::CreateMatrixDescriptor(
        proj_q->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    auto mat_dim_proj_k = math::CreateMatrixDescriptor(
        proj_k->dims(), /* num_flatten_cols */ 0, /* trans */ true);
    if (attn_bias) {
      TensorCopy(*attn_bias, context.GetPlace(), dot_product);
      blas.MatMul(*proj_q, mat_dim_proj_q, *proj_k, mat_dim_proj_k, T(1),
                  dot_product, T(1));
    } else {
      blas.MatMul(*proj_q, mat_dim_proj_q, *proj_k, mat_dim_proj_k, T(1),
                  dot_product, T(0));
    }
    dot_product->Resize({batch_size * n_head * src_seq_len, trg_seq_len});
    if (std::is_same<DeviceContext, platform::CPUDeviceContext>::value) {
      math::SoftmaxFunctor<DeviceContext, T>()(
          context.template device_context<DeviceContext>(), dot_product,
          attn_weight);
    } else {
#ifdef PADDLE_WITH_CUDA
      math::SoftmaxCUDNNFunctor<T>()(
          context.template device_context<platform::CUDADeviceContext>(),
          dot_product, attn_weight);
#else
      math::SoftmaxFunctor<DeviceContext, T>()(
          context.template device_context<DeviceContext>(), dot_product,
          attn_weight);
#endif
    }
    auto dropout_prob = context.Attr<float>("dropout_prob");
    if (dropout_prob != 0) {
      platform::ForRange<DeviceContext> for_range(dev_ctx,
                                                  attn_weight->numel());
      std::random_device rnd;
      int seed =
          context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();
      if (context.Attr<bool>("is_test")) {
        for_range(ScaleFunctor<T>(attn_weight->data<T>(),
                                  attn_weight->data<T>(), scale));
      } else {
        for_range(DropoutFunctor<DeviceContext, T, uint8_t>(
            attn_weight->data<T>(), attn_weight->data<T>(),
            mask->data<uint8_t>(), dropout_prob, seed));
      }
    }
    auto mat_dim_attn_weight = math::CreateMatrixDescriptor(
        attn_weight->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    auto mat_dim_proj_v = math::CreateMatrixDescriptor(
        proj_v->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    blas.MatMul(*attn_weight, mat_dim_attn_weight, *proj_v, mat_dim_proj_v,
                T(1), attn_context, T(0));

    // combine_heads
    attn_context->Resize({batch_size, n_head, src_seq_len, d_value});
    trans(dev_ctx, *attn_context, attn_context, axis);
    attn_context->Resize({batch_size, src_seq_len, n_head * d_value});

    auto mat_dim_attn_context = math::CreateMatrixDescriptor(
        attn_context->dims(), /* num_flatten_cols */ 2, /* trans */ false);
    auto mat_dim_weight_out = math::CreateMatrixDescriptor(
        weight_q->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    // linear projection
    blas.MatMul(*attn_context, mat_dim_attn_context, *weight_out,
                mat_dim_weight_out, T(1), out, T(0));

    // reshape outputs since they have been reshaped for calculations
    proj_q->Resize({batch_size, src_seq_len, n_head * d_key});
    proj_k->Resize({batch_size, trg_seq_len, n_head * d_key});
    proj_v->Resize({batch_size, trg_seq_len, n_head * d_value});
    dot_product->Resize({batch_size, n_head, src_seq_len, trg_seq_len});
    // attn_weight->Resize({batch_size, n_head, src_seq_len, trg_seq_len});
    // mask->Resize({batch_size, n_head, src_seq_len, trg_seq_len});
    // attn_context->Resize({batch_size, src_seq_len, n_head * d_value});
    // out->Resize({batch_size, src_seq_len, d_model});
  }
};

template <typename DeviceContext, typename T>
class MultiHeadAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    // auto &dev_ctx = context.template device_context<DeviceContext>();

    // auto *q = context.Input<framework::Tensor>("Q");
    // auto *k = context.Input<framework::Tensor>("K");
    // auto *v = context.Input<framework::Tensor>("V");
    // auto *weight_q = context.Input<framework::Tensor>("Weight_Q");
    // auto *weight_k = context.Input<framework::Tensor>("Weight_K");
    // auto *weight_v = context.Input<framework::Tensor>("Weight_V");
    // auto *proj_q = context.Input<framework::Tensor>("Proj_Q");
    // auto *proj_k = context.Input<framework::Tensor>("Proj_K");
    // auto *proj_v = context.Input<framework::Tensor>("Proj_V");
    // auto *cache_k = context.Input<framework::Tensor>("Cache_K");
    // auto *cache_v = context.Input<framework::Tensor>("Cache_V");
    // auto *concat_k = context.Input<framework::Tensor>("Concat_K");
    // auto *concat_v = context.Input<framework::Tensor>("Concat_V");
    // auto *dot_product = context.Input<framework::Tensor>("Dot_Product");
    // auto *attn_bias = context.Input<framework::Tensor>("Attn_Bias");
    // auto *attn_weight = context.Input<framework::Tensor>("Attn_Weight");
    // auto *mask = context.Input<framework::Tensor>("Mask");
    // auto *attn_context = context.Input<framework::Tensor>("Attn_Context");
    // auto *weight_out = context.Input<framework::Tensor>("Weight_Out");
    // auto *out = context.Input<framework::Tensor>("Out");
    // auto *dx =
    // context.Output<framework::Tensor>(framework::GradVarName("X"));
    // auto *dy =
    // context.Output<framework::Tensor>(framework::GradVarName("Y"));

    // proj_q->mutable_data<T>(context.GetPlace());
    // proj_k->mutable_data<T>(context.GetPlace());
    // proj_v->mutable_data<T>(context.GetPlace());
    // dot_product->mutable_data<T>(context.GetPlace());
    // mask->mutable_data<uint8_t>(context.GetPlace());
    // attn_context->mutable_data<T>(context.GetPlace());
    // out->mutable_data<T>(context.GetPlace());

    // auto scale = static_cast<T>(context.Attr<float>("scale"));
    // auto n_head = context.Attr<int>("n_head");
    // auto batch_size = q->dims()[0];
    // auto src_seq_len = q->dims()[1];
    // auto trg_seq_len = k->dims()[1];
    // auto d_model = q->dims()[2];
    // auto d_key = weight_k->dims()[1] / n_head;
    // auto d_value = weight_v->dims()[1] / n_head;
  }
};

}  // namespace operators
}  // namespace paddle
