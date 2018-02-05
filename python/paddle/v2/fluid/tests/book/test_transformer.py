#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.executor import Executor

src_vocab_size = 10000
trg_vocab_size = 10000
src_pad_idx = src_vocab_size
trg_pad_idx = trg_vocab_size

n_layer = 2  # 6
n_head = 2  # 8
d_model = 512
d_inner_hid = 1024
d_key = 64
d_value = 64
dropout = 0.1
batch_size = 1  # 64
max_length = 50

transformer_output = None


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         num_heads=1,
                         dropout_rate=0.):
    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, num_heads, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.

        Args:
            queries(Tensor): a 3-D input Tensor.
            keys(Tensor): a 3-D input Tensor.
            values(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads. Linearly project the inputs
                            ONLY when num_heads > 1.

        Returns:
            Tensor: linearly projected output Tensors: queries', keys' and
                    values'. They have the same shapes with queries, keys and
                    values.
        """

        q = layers.fc(input=queries,
                      size=d_key * num_heads,
                      bias_attr=False,
                      num_flatten_dims=2)
        k = layers.fc(input=keys,
                      size=d_key * num_heads,
                      bias_attr=False,
                      num_flatten_dims=2)
        v = layers.fc(input=values,
                      size=d_value * num_heads,
                      bias_attr=False,
                      num_flatten_dims=2)
        return q, k, v

    def __split_heads(x, num_heads):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions.

        Args:
            x(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads.

        Returns:
            Tensor: a Tensor with shape [..., n, m/num_heads], where m is size
                    of the last dimension of x.
        """
        if num_heads == 1:
            return x

        hidden_size = x.shape[-1]
        # reshape the 3-D input: [batch_size, max_sequence_length, hidden_dim]
        # into a 4-D output:
        # [batch_size, max_sequence_length, num_heads, hidden_size_per_head].
        # reshaped = layers.reshape(
        #     x=x,
        #     shape=list(x.shape[:-1]) + [num_heads, hidden_size // num_heads])
        # TODO: Decouple the program desc with batch_size
        reshaped = layers.reshape(
            x=x, shape=[batch_size, -1, num_heads, hidden_size // num_heads])

        # permuate the dimensions into:
        # [batch_size, num_heads, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Reshape the last two dimensions of inpunt tensor x so that it becomes
        one dimension.

        Args:
            x(Tensor): a 4-D input Tensor with shape
                       [bs, num_heads, max_sequence_length, hidden_dim].

        Returns:
            Tensor: a Tensor with shape
                    [bs, max_sequence_length, num_heads * hidden_dim].
        """

        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # return layers.reshape(
        #     x=trans_x,
        #     shape=map(int, [
        #         trans_x.shape[0], trans_x.shape[1],
        #         trans_x.shape[2] * trans_x.shape[3]
        #     ]))
        # TODO: Decouple the program desc with batch_size
        return layers.reshape(
            x=trans_x,
            shape=map(int,
                      [batch_size, -1, trans_x.shape[2] * trans_x.shape[3]]))

    q, k, v = __compute_qkv(queries, keys, values, num_heads, d_key, d_value)

    q = __split_heads(q, num_heads)
    k = __split_heads(k, num_heads)
    v = __split_heads(v, num_heads)

    key_dim_per_head = keys.shape[-1] // num_heads
    print keys.shape[-1], num_heads
    scaled_q = layers.scale(x=q, scale=key_dim_per_head**-0.5)
    product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
    global transformer_output
    if transformer_output == 1:
        transformer_output = product
    # weights = layers.reshape(
    #     x=layers.reshape(
    #         x=layers.elementwise_add(x=product, y=attn_bias),
    #         shape=[-1, product.shape[-1]],
    #         act="softmax"),
    #     shape=product.shape)
    # TODO: Optimize the shape in reshape_op or softmax_op.
    # The shape of produce inferred in compile-time is not the actual shape in
    # run-time and cann't be used to set the attribute of reshape_op.
    # softmax_op only support 2D tensor currently.
    def __softmax(x):
        exp_out = layers.exp(x=x)
        sum_out = layers.reduce_sum(x, dim=-1, keep_dim=False)
        return layers.elementwise_div(x=exp_out, y=sum_out, axis=0)

    # print product.dtype, attn_bias.dtype, product.type, attn_bias.type
    weights = __softmax(layers.elementwise_add(x=product, y=attn_bias))
    # weights = __softmax(product)
    # global transformer_output
    # if transformer_output == 1:
    #     transformer_output = weights
    if dropout_rate:
        weights = layers.dropout(
            weights, dropout_prob=dropout_rate, is_test=False)
    ctx_multiheads = layers.matmul(weights, v)
    out = __combine_heads(ctx_multiheads)
    proj_out = layers.fc(input=out,
                         size=d_model,
                         bias_attr=False,
                         num_flatten_dims=2)
    # global transformer_output
    # if transformer_output is None:
    #     transformer_output = proj_out
    return proj_out


def pointwise_feed_forward(x, d_inner_hid, d_hid):
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       bias_attr=False,
                       num_flatten_dims=2,
                       act="relu")
    out = layers.fc(input=hidden,
                    size=d_hid,
                    bias_attr=False,
                    num_flatten_dims=2)
    return out


def post_process_layer(prev_out, out, process_cmd, dropout=0.):
    for cmd in process_cmd:
        if cmd == "a":
            out += prev_out
        elif cmd == "n":
            out = layers.layer_norm(out)
        elif cmd == "d":
            if dropout:
                out = layers.dropout(out, dropout_prob=dropout, is_test=False)
    return out


def prepare_encoder(src_word, src_pos, src_vocab_size, src_emb_dim, src_pad_idx,
                    max_length):
    src_word_emb = layers.embedding(
        src_word, size=[src_vocab_size, src_emb_dim], padding_idx=src_pad_idx)
    sor_pos_enc = layers.embedding(src_pos, size=[max_length, src_emb_dim])
    enc_input = src_word_emb + sor_pos_enc
    # TODO: Decouple the program desc with batch_size
    enc_input = layers.reshape(x=enc_input, shape=[batch_size, -1, src_emb_dim])
    return enc_input


prepare_decoder = prepare_encoder


def encoder_layer(enc_input, attn_bias, n_head, d_key, d_value, d_model,
                  d_inner_hid, dropout):
    attn_output = multi_head_attention(enc_input, enc_input, enc_input,
                                       attn_bias, d_key, d_value, d_model,
                                       n_head, dropout)
    attn_output = post_process_layer(enc_input, attn_output, 'da', dropout)
    # global transformer_output
    # if transformer_output == 1:
    #     transformer_output = attn_output
    ffd_output = pointwise_feed_forward(attn_output, d_inner_hid, d_model)
    output = post_process_layer(attn_output, ffd_output, 'da', dropout)
    return output


def encoder(enc_input, attn_bias, n_layer, n_head, d_key, d_value, d_model,
            d_inner_hid, dropout):
    for i in range(n_layer):
        global transformer_output
        if i == 1:  #transformer_output is None and i == 1:
            transformer_output = 1
        enc_output = encoder_layer(enc_input, attn_bias, n_head, d_key, d_value,
                                   d_model, d_inner_hid, dropout)
        enc_input = enc_output
    return enc_output


def decoder_layer(dec_input, enc_output, slf_attn_bias, dec_enc_attn_bias,
                  n_head, d_key, d_value, d_model, d_inner_hid, dropout):
    slf_attn_output = multi_head_attention(dec_input, dec_input, dec_input,
                                           slf_attn_bias, d_key, d_value,
                                           d_model, n_head, dropout)
    slf_attn_output = post_process_layer(dec_input, slf_attn_output, 'da',
                                         dropout)
    enc_attn_output = multi_head_attention(slf_attn_output, enc_output,
                                           enc_output, dec_enc_attn_bias, d_key,
                                           d_value, d_model, n_head, dropout)
    enc_attn_output = post_process_layer(slf_attn_output, enc_attn_output, 'da',
                                         dropout)
    ffd_output = pointwise_feed_forward(enc_attn_output, d_inner_hid, d_model)
    dec_output = post_process_layer(enc_attn_output, ffd_output, 'da', dropout)
    return dec_output


def decoder(dec_input, enc_output, dec_slf_attn_bias, dec_enc_attn_bias,
            n_layer, n_head, d_key, d_value, d_model, d_inner_hid, dropout):
    for i in range(n_layer):
        dec_output = decoder_layer(dec_input, enc_output, dec_slf_attn_bias,
                                   dec_enc_attn_bias, n_head, d_key, d_value,
                                   d_model, d_inner_hid, dropout)
        dec_input = dec_output
    return dec_output


def transformer(src_vocab_size, trg_vocab_size, max_length, n_layer, n_head,
                d_key, d_value, d_model, d_inner_hid, dropout):
    # The shapes here only act as placeholder and are set to confirm the
    # success of infer-shape in compile time.
    # The actual shape of src_word is:
    # [batch_size * max_src_length_in_batch, 1].
    src_word = layers.data(
        name='src_word',
        shape=[batch_size * max_length, 1],
        dtype='int64',
        append_batch_size=False)
    # The actual shape of src_pos is:
    # [batch_size * max_src_length_in_batch, 1].
    src_pos = layers.data(
        name='src_pos',
        shape=[batch_size * max_length, 1],
        dtype='int64',
        append_batch_size=False)
    # The actual shape of trg_word is:
    # [batch_size * max_trg_length_in_batch, 1].
    trg_word = layers.data(
        name='trg_word',
        shape=[batch_size * max_length, 1],
        dtype='int64',
        append_batch_size=False)
    # The actual shape of trg_pos is:
    # [batch_size * max_trg_length_in_batch, 1].
    trg_pos = layers.data(
        name='trg_pos',
        shape=[batch_size * max_length, 1],
        dtype='int64',
        append_batch_size=False)
    # The actual shape of src_slf_attn_bias is:
    # [batch_size, n_head, max_src_length_in_batch, max_src_length_in_batch].
    # This is used to avoid attention on paddings.
    src_slf_attn_bias = layers.data(
        name='src_slf_attn_bias',
        shape=[batch_size, n_head, max_length, max_length],
        dtype='float32',
        append_batch_size=False)
    # The actual shape of trg_slf_attn_bias is:
    # [batch_size, n_head, max_trg_length_in_batch, max_trg_length_in_batch].
    # This is used to avoid attention on paddings and subsequent words.
    trg_slf_attn_bias = layers.data(
        name='trg_slf_attn_bias',
        shape=[batch_size, n_head, max_length, max_length],
        dtype='float32',
        append_batch_size=False)
    # The actual shape of trg_src_attn_bias is:
    # [batch_size, n_head, max_trg_length_in_batch, max_src_length_in_batch].
    # This is used to avoid attention on paddings.
    trg_src_attn_bias = layers.data(
        name='trg_src_attn_bias',
        shape=[batch_size, n_head, max_length, max_length],
        dtype='float32',
        append_batch_size=False)

    enc_input = prepare_encoder(src_word, src_pos, src_vocab_size, d_model,
                                src_pad_idx, max_length)
    enc_output = encoder(enc_input, src_slf_attn_bias, n_layer, n_head, d_key,
                         d_value, d_model, d_inner_hid, dropout)

    dec_input = prepare_decoder(trg_word, trg_pos, trg_vocab_size, d_model,
                                trg_pad_idx, max_length)
    dec_output = decoder(dec_input, enc_output, trg_slf_attn_bias,
                         trg_src_attn_bias, n_layer, n_head, d_key, d_value,
                         d_model, d_inner_hid, dropout)

    predict = layers.reshape(
        x=layers.fc(input=dec_output,
                    size=trg_vocab_size,
                    bias_attr=False,
                    num_flatten_dims=2),
        shape=[-1, trg_vocab_size],
        act="softmax")
    # The actual shape of gold is:
    # [batch_size * max_trg_length_in_batch, 1].
    gold = layers.data(
        name='lbl_word',
        shape=[batch_size * max_length, 1],
        dtype='int64',
        append_batch_size=False)
    cost = layers.cross_entropy(input=predict, label=gold)
    avg_cost = layers.mean(x=cost)
    global transformer_output
    if transformer_output is None:
        transformer_output = avg_cost
    return transformer_output  # avg_cost


def prepare_batch_input(insts, src_pad_idx, trg_pad_idx, max_length, n_head,
                        place):
    input_dict = {}

    def pad_batch_data(insts,
                       pad_idx,
                       is_target=False,
                       return_pos=True,
                       return_attn_bias=True,
                       return_max_len=True):
        return_list = []
        max_len = max(len(inst) for inst in insts)
        # max_len = min(max(len(inst) for inst in insts), max_length)
        # insts = [inst for inst in insts if len(inst) <= max_len]
        inst_data = np.array(
            [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_data.astype("int64").reshape([-1, 1])]
        if return_pos:
            inst_pos = np.array([[
                pos_i + 1 if w_i != pad_idx else 0
                for pos_i, w_i in enumerate(inst)
            ] for inst in inst_data])

            return_list += [inst_pos.astype("int64").reshape([-1, 1])]
        if return_attn_bias:
            if is_target:
                # This is used to avoid attention on paddings and subsequent words.
                slf_attn_bias_data = np.ones((inst_data.shape[0], max_len,
                                              max_len)).astype("float32")
                slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                    [-1, 1, max_len, max_len])
                slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                             [1, n_head, 1, 1]) * [-1e2]
            else:
                # This is used to avoid attention on paddings.
                slf_attn_bias_data = np.array(
                    [[0] * len(inst) + [-1e2] * (max_len - len(inst))
                     for inst in insts]).astype("float32")
                slf_attn_bias_data = np.tile(
                    slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                    [1, n_head, max_len, 1])
            return_list += [slf_attn_bias_data]
        if return_max_len:
            return_list += [max_len]
        return return_list if len(return_list) > 1 else return_list[0]

    def data_to_tensor(data_list, name_list, input_dict, place):
        assert len(data_list) == len(name_list)
        for i in range(len(name_list)):
            tensor = core.LoDTensor()
            tensor.set(data_list[i], place)
            input_dict[name_list[i]] = tensor

    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, is_target=False)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")
    lbl_word = pad_batch_data([inst[2] for inst in insts], trg_pad_idx, False,
                              False, False, False)

    data_to_tensor([src_word, src_pos, src_slf_attn_bias],
                   ['src_word', 'src_pos', 'src_slf_attn_bias'], input_dict,
                   place)
    data_to_tensor(
        [trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, lbl_word], [
            'trg_word', 'trg_pos', 'trg_slf_attn_bias', 'trg_src_attn_bias',
            'lbl_word'
        ], input_dict, place)

    print src_slf_attn_bias.shape, trg_src_attn_bias.shape
    return input_dict


def main():
    avg_cost = transformer(src_vocab_size, trg_vocab_size, max_length, n_layer,
                           n_head, d_key, d_value, d_model, d_inner_hid,
                           dropout)

    # optimizer = fluid.optimizer.Adagrad(learning_rate=1e-4)
    # optimizer.minimize(avg_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt16.train(src_vocab_size, src_vocab_size),
            buf_size=1000),
        batch_size=batch_size)

    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(framework.default_startup_program())
    # print(framework.default_main_program())
    # exit(0)

    batch_id = 0
    for pass_id in xrange(2):
        for data in train_data():
            data_input = prepare_batch_input(data, src_pad_idx, trg_pad_idx,
                                             max_length, n_head, place)
            outs = exe.run(framework.default_main_program(),
                           feed=data_input,
                           fetch_list=[avg_cost])
            avg_cost_val = np.array(outs[0])
            print('pass_id=' + str(pass_id) + ' batch=' + str(batch_id) +
                  " avg_cost=" + str(avg_cost_val))
            if batch_id >= 0:
                exit(0)
            batch_id += 1


if __name__ == '__main__':
    main()
