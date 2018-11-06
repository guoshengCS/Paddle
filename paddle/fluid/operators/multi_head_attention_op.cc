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

#include "paddle/fluid/operators/multi_head_attention_op.h"

namespace paddle {
namespace operators {

class MultiHeadAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("Q"),
                   "Input(Q) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(context->HasInput("K"),
                   "Input(K) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(context->HasInput("V"),
                   "Input(V) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasInput("Weight_Q"),
        "Input(Weight_Q) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasInput("Weight_K"),
        "Input(Weight_K) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasInput("Weight_V"),
        "Input(Weight_V) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasInput("Weight_Out"),
        "Input(Weight_Out) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasOutput("Proj_Q"),
        "Output(Proj_Q) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasOutput("Proj_K"),
        "Output(Proj_K) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasOutput("Proj_V"),
        "Output(Proj_V) of MultiHeadAttentionOp should not be null.");
    // PADDLE_ENFORCE(
    //     context->HasOutput("Trans_Q"),
    //     "Output(Trans_Q) of MultiHeadAttentionOp should not be null.");
    // PADDLE_ENFORCE(
    //     context->HasOutput("Trans_K"),
    //     "Output(Trans_K) of MultiHeadAttentionOp should not be null.");
    // PADDLE_ENFORCE(
    //     context->HasOutput("Trans_V"),
    //     "Output(Trans_V) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasOutput("Dot_Product"),
        "Output(Dot_Product) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasOutput("Attn_Weight"),
        "Output(Attn_Weight) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(context->HasOutput("Mask"),
                   "Output(Mask) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(
        context->HasOutput("Attn_Context"),
        "Output(Attn_Context) of MultiHeadAttentionOp should not be null.");
    // PADDLE_ENFORCE(
    //     context->HasOutput("Trans_Context"),
    //     "Output(Trans_Context) of MultiHeadAttentionOp should not be null.");
    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "Output(Out) of MultiHeadAttentionOp should not be null.");
    auto dim_q = context->GetInputDim("Q");
    auto dim_k = context->GetInputDim("K");
    auto dim_v = context->GetInputDim("V");
    auto dim_weight_k = context->GetInputDim("Weight_K");
    auto dim_weight_v = context->GetInputDim("Weight_V");
    auto batch_size = dim_q[0];
    auto src_seq_len = dim_q[1];
    auto trg_seq_len = dim_k[1];
    auto d_model = dim_q[2];
    auto n_head = context->Attrs().Get<int>("n_head");
    auto d_key = dim_weight_k[1] / n_head;
    auto d_value = dim_weight_v[1] / n_head;

    if (context->HasInput("Cache_K") && context->HasInput("Cache_V")) {
      auto dim_cacke_k = context->GetInputDim("Cache_K");
      trg_seq_len += dim_cacke_k[1];
      context->SetOutputDim("Concat_K",
                            {batch_size, trg_seq_len, dim_weight_k[1]});
      context->SetOutputDim("Concat_V",
                            {batch_size, trg_seq_len, dim_weight_v[1]});
    }
    context->SetOutputDim("Proj_Q", {batch_size, n_head, src_seq_len, d_key});
    context->SetOutputDim("Proj_K", {batch_size, n_head, trg_seq_len, d_key});
    context->SetOutputDim("Proj_V", {batch_size, n_head, trg_seq_len, d_value});
    // context->SetOutputDim("Trans_Q", {batch_size, n_head, src_seq_len,
    // d_key});
    // context->SetOutputDim("Trans_K", {batch_size, n_head, trg_seq_len,
    // d_key});
    // context->SetOutputDim("Trans_V",
    //                       {batch_size, n_head, trg_seq_len, d_value});
    context->SetOutputDim("Dot_Product",
                          {batch_size * n_head * src_seq_len, trg_seq_len});
    context->SetOutputDim("Attn_Weight",
                          {batch_size * n_head * src_seq_len, trg_seq_len});
    context->SetOutputDim("Mask",
                          {batch_size, n_head, src_seq_len, trg_seq_len});
    context->SetOutputDim("Attn_Context",
                          {batch_size, src_seq_len, n_head, d_value});
    context->SetOutputDim("Out", {batch_size, src_seq_len, d_model});
  }
};

class MultiHeadAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Q", "The first input of MatMul op");
    AddInput("K", "The second input of MatMul op");
    AddInput("V", "The second input of MatMul op");
    AddInput("Weight_Q", "The first input of MatMul op");
    AddInput("Weight_K", "The first input of MatMul op");
    AddInput("Weight_V", "The first input of MatMul op");
    AddOutput("Proj_Q", "The output of MatMul op").AsIntermediate();
    AddOutput("Proj_K", "The output of MatMul op").AsIntermediate();
    AddOutput("Proj_V", "The output of MatMul op").AsIntermediate();
    AddInput("Cache_K", "The second input of MatMul op").AsDispensable();
    AddInput("Cache_V", "The second input of MatMul op").AsDispensable();
    AddOutput("Concat_K", "The second input of MatMul op");
    AddOutput("Concat_V", "The second input of MatMul op");
    // AddOutput("Trans_Q", "The output of MatMul op").AsIntermediate();
    // AddOutput("Trans_K", "The output of MatMul op").AsIntermediate();
    // AddOutput("Trans_V", "The output of MatMul op").AsIntermediate();
    AddOutput("Dot_Product", "The output of MatMul op").AsIntermediate();
    AddInput("Attn_Bias", "The second input of MatMul op").AsDispensable();
    AddOutput("Attn_Weight", "The output of MatMul op").AsIntermediate();
    AddOutput("Mask", "The random sampled dropout mask.").AsIntermediate();
    AddOutput("Attn_Context", "The output of MatMul op").AsIntermediate();
    // AddOutput("Trans_Context", "The output of MatMul op").AsIntermediate();
    AddInput("Weight_Out", "The first input of MatMul op");
    AddOutput("Out", "The output of MatMul op");
    AddAttr<float>("scale", "Probability of setting units to zero.");
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
MultiHeadAttention Operator.

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
