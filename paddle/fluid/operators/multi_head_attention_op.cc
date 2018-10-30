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

#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct DropoutFunctor {
  inline DropoutFunctor(const T *in, T *out, bool is_test, int seed)
      : in_(in), out_(out), is_test_(is_test), seed_(seed) {}

  HOSTDEVICE inline T operator()(int64_t idx) const {}

  const T *in_;
  T *out_;
  bool is_test_;
  int seed_;
};

template <typename T>
struct ScaleFunctor {
  inline ScaleFunctor(const T *in, T *out, T scale)
      : in_(in), out_(out), scale_(scale) {}

  HOSTDEVICE inline T operator()(int64_t idx) const {}

  const T *in_;
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
    auto *cache_k = context.Input<framework::Tensor>("Cache_K");
    auto *cache_v = context.Input<framework::Tensor>("Cache_V");
    auto *attn_bias = context.Input<framework::Tensor>("Attn_Bias");
    auto *attn_weight = context.Output<framework::Tensor>("Attn_weight");
    auto *proj_q = context.Output<framework::Tensor>("Proj_Q");
    auto *proj_k = context.Output<framework::Tensor>("Proj_K");
    auto *proj_v = context.Output<framework::Tensor>("Proj_V");
    auto *out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    auto dropout_prob = context.Attr<float>("dropout_prob");
    auto is_test = context.Attr<bool>("is_test");
    auto fix_seed = context.Attr<bool>("fix_seed");
    auto seed = context.Attr<int>("seed");
    auto n_head = context.Attr<int>("n_head");

    T d_key = w_k->dims()[1] / n_head;
    T scale = 1 / sqrt(d_key);
    // auto z_dim = z->dims();
    // if (z_dim.size() != 2) {
    //   z->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    // }
    auto = math::MatrixDescriptor();
    auto blas = math::GetBlas<DeviceContext, T>(context);
    // compute_qkv
    auto mat_dim_q = math::CreateMatrixDescriptor(
        q->dims(), /* num_flatten_cols */ 2, /* trans */ false);
    auto mat_dim_w_q = math::CreateMatrixDescriptor(
        w_q->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    auto mat_dim_k = math::CreateMatrixDescriptor(
        k->dims(), /* num_flatten_cols */ 2, /* trans */ false);
    auto mat_dim_w_k = math::CreateMatrixDescriptor(
        w_k->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    auto mat_dim_v = math::CreateMatrixDescriptor(
        k->dims(), /* num_flatten_cols */ 2, /* trans */ false);
    auto mat_dim_w_v = math::CreateMatrixDescriptor(
        w_k->dims(), /* num_flatten_cols */ 0, /* trans */ false);
    blas.MatMul(*q, mat_dim_q, *weight_q, mat_dim, scale, proj_q, T(0));
    blas.MatMul(*k, mat_dim, *weight_k, mat_dim, T(1), proj_k, T(0));
    blas.MatMul(*v, mat_dim, *weight_v, mat_dim, T(1), proj_v, T(0));
    // split_heads

    math::Transpose<DeviceContext, T, 4> trans;
    std::vector<int> axis{0, 2, 1, 3};
    trans(dev_ctx, *proj_q, proj_q, axis);
    trans(dev_ctx, *proj_k, proj_k, axis);
    trans(dev_ctx, *proj_v, proj_v, axis);

    // dot_product_attention
    TensorCopy(*attn_bias, context.GetPlace(), attn_weight);
    blas.MatMul(*q_out, mat_dim_q, *k_out, mat_dim, T(1), attn_weight, T(1));
    if (std::is_same<DeviceContext, platform::CPUDeviceContext>::value) {
      math::SoftmaxFunctor<DeviceContext, T>()(
          context.template device_context<DeviceContext>(), &X_2d, &Out_2d);
    } else {
      math::SoftmaxCUDNNFunctor<T>()(
          context.template device_context<platform::CUDADeviceContext>(),
          &flattened_x, &flattened_out);
    }
    platform::ForRange<DeviceContext> for_range(dev_ctx, attn_weight->numel());
    if (is_test) {
      for_range(ScaleFunctor());
    } else {
      for_range(DropoutFunctor());
    }
    blas.MatMul(*q_out, mat_dim_q, *k_out, mat_dim, T(1), attn_weight, T(1));
    // combine_heads
    trans(dev_ctx, *proj_q, proj_q, axis);
    // linear projection
    blas.MatMul(*v, mat_dim, *w_v, mat_dim, scale, out, T(0));
  }
};

template <typename DeviceContext, typename T>
class MultiHeadAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {}
};

class MultiHeadAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {}
};

class MultiHeadAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Q", "The first input of MatMul op");
    AddInput("K", "The second input of MatMul op").AsDispensable();
    AddInput("V", "The second input of MatMul op").AsDispensable();
    AddInput("Weight_Q", "The first input of MatMul op");
    AddInput("Weight_K", "The first input of MatMul op");
    AddInput("Weight_V", "The first input of MatMul op");
    AddInput("Weight_Out", "The first input of MatMul op");
    AddInput("Attn_bias", "The second input of MatMul op").AsDispensable();
    AddInput("Cache_K", "The second input of MatMul op").AsDispensable();
    AddInput("Cache_V", "The second input of MatMul op").AsDispensable();
    AddOutput("Mask", "The random sampled dropout mask.").AsIntermediate();
    AddOutput("Attn_Weight", "The output of MatMul op").AsIntermediate();
    AddOutput("Proj_Q", "The output of MatMul op").AsIntermediate();
    AddOutput("Proj_K", "The output of MatMul op").AsIntermediate();
    AddOutput("Proj_V", "The output of MatMul op").AsIntermediate();
    AddOutput("Out", "The output of MatMul op");
    AddAttr<int>("n_head",
                 R"DOC(If true, use the transpose of `X`.
        )DOC");
    AddAttr<float>("dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE(drop_p >= 0.0f && drop_p <= 1.0f,
                         "'dropout_prob' must be between 0.0 and 1.0.");
        });
    AddAttr<bool>("is_test", "True if in test phase.").SetDefault(false);
    AddAttr<bool>("fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug that always the same output units "
                  "will be dropped.")
        .SetDefault(false);
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0);
    AddComment(R"DOC(
MatMul Operator.

This operator is used to perform (batched) matrix multiplication
over the last two dimensions of the input tensors `X` and `Y`.

)DOC");
  }
};

class MultiHeadAttentionGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(multi_head_attention, ops::MultiHeadAttentionOp,
                  ops::MultiHeadAttentionOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(multi_head_attention_grad, ops::MultiHeadAttentionGradOp);
REGISTER_OP_CPU_KERNEL(
    multi_head_attention,
    ops::MultiHeadAttentionKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MultiHeadAttentionKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    multi_head_attention_grad,
    ops::MultiHeadAttentionGradKernel<paddle::platform::CPUDeviceContext,
                                      float>,
    ops::MultiHeadAttentionGradKernel<paddle::platform::CPUDeviceContext,
                                      double>);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    multi_head_attention,
    ops::MultiHeadAttentionKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MultiHeadAttentionKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    multi_head_attention_grad,
    ops::MultiHeadAttentionGradKernel<paddle::platform::CUDADeviceContext,
                                      float>,
    ops::MultiHeadAttentionGradKernel<paddle::platform::CUDADeviceContext,
                                      double>);
#endif
