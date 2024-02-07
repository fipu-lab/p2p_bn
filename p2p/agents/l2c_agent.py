from p2p.agents.sync_agent import *
from models.abstract_model import _default_weights, weights_average
import numpy as np
import copy


# Learning to Collaborate in Decentralized Learning of Personalized Models
# Authors: Li, Shuangtong
#          Zhou, Tianyi
#          Tian, Xinmei
#          Tao, Dachengk


class L2CAgent(SyncAgent):
    def __init__(self, k_0=90, t_0=10, agg='avg', **kwargs):
        super(L2CAgent, self).__init__(**kwargs)
        assert agg in ['grad', 'avg']
        self.k_0 = k_0
        self.t_0 = t_0
        self.agg = agg
        self.iteration = 0
        self.weights_delta = None
        self.new_weights = None
        self.mixing_net = None
        self.mixing_weight_mask = []
        self.mixing_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, decay=0.01)
        self.hist['mixing_weight_mask'] = []

    def start(self):
        if 0 < self.k_0 < 1:
            self.k_0 = int(self.k_0 * self.graph.nodes_num)
        assert self.k_0 < self.graph.nodes_num - 1, "Cannot remove more peers that there are agents"

        self.mixing_net = GCN(self.graph.nodes_num)
        # assert self.graph.graph_type == "complete", "Graph type not set to complete"
        self.new_weights = self.get_model_weights()
        return super(L2CAgent, self).start()

    def train_fn(self):
        self.set_model_weights(self.new_weights)
        self.iteration += 1
        res = super(L2CAgent, self).train_fn()
        self.weights_delta = tf.nest.map_structure(lambda w1, w2: w1 - w2, self.new_weights, self.get_model_weights())
        return res

    def aggregation(self):
        # self can also be "sampled"
        peers = [p for p in self.graph.nodes if p.id in self.mixing_weight_mask or len(self.mixing_weight_mask) == 0]
        for peer in peers:
            if peer.id != self.id:
                self.receive_message(peer)
        self.update_mixing_weights(peers)

    def update_mixing_weights(self, peers):
        assert len(_default_weights) == 1, "Not implemented yet to have more than one model with initial weights"

        grad_group = [[] for _ in range(self.graph.nodes_num)]
        model_update_group = [[] for _ in range(len(grad_group))]
        for p in peers:
            grad_group[p.id] = p.weights_delta
            model_update_group[p.id] = tf.nest.map_structure(lambda w1, w2: w1 - w2,
                                                             p.get_model_weights(), list(_default_weights.values())[0])

        losses = np.zeros(len(grad_group))
        losses[[p.id for p in peers]] = [self.eval_model_loss(p.model, self.val) for p in peers]
        mixing_weights, y = calc_optimal_mixing_weights(losses)

        mixing_weight_pred = self.update_mixing_model(mixing_weights, model_update_group)
        final_mixing_weight = mixing_weight_process(mixing_weight_pred, y, self.mixing_weight_mask)
        if self.agg == 'grad':
            if len(self.mixing_weight_mask) > 0:
                grad_group = [grad_group[i] for i in self.mixing_weight_mask]
            new_grad = combine_weights_deltas(grad_group, final_mixing_weight)
            self.new_weights = tf.nest.map_structure(lambda w1, w2: w1 - w2, self.new_weights, new_grad)
        else:
            self.new_weights = weights_average([peer.get_model_weights() for peer in peers], final_mixing_weight)
        self.cut_peers(mixing_weights)

    def update_mixing_model(self, mixing_weights, model_update_group):
        with tf.GradientTape() as tape:
            masked_weights = tf.convert_to_tensor(mixing_weight_to_masked(mixing_weights, self.mixing_weight_mask))
            mixing_weight_obj = tf.reshape(masked_weights, [-1])

            mixing_weight_pred = self.mixing_net([model_update_group, self.mixing_weight_mask])
            log_pred = -tf.math.log(tf.reshape(mixing_weight_pred, [-1]) + 1e-10)
            l_g_meta = tf.reduce_sum(tf.cast(log_pred, mixing_weight_obj.dtype) * mixing_weight_obj)

        gradients = tape.gradient(l_g_meta, self.mixing_net.trainable_variables)
        self.mixing_optimizer.apply_gradients(zip(gradients, self.mixing_net.trainable_variables))
        return mixing_weight_pred

    def cut_peers(self, mixing_weights):
        # Cut K_0 peers from communication matrix
        if self.iteration == self.t_0:
            inds = np.argsort(mixing_weights)
            inds = inds[inds != self.id]
            self.mixing_weight_mask = np.sort(np.concatenate([inds[self.k_0-1:], [self.id]]))

    def sync_parameters(self):
        self.aggregation()

    def update_parameters(self):
        self.hist['mixing_weight_mask'].append(self.mixing_weight_mask)


class GCN(tf.keras.Model):
    def __init__(self, size):
        super(GCN, self).__init__()
        self.model_weight = tf.Variable(tf.zeros(size), trainable=True)

    def call(self, inputs, training=None, mask=None):
        _, mixing_weight_mask = inputs[0], inputs[1]

        all_indices = tf.range(self.model_weight.shape[0], dtype=tf.int32)
        indices = tf.cond(len(mixing_weight_mask) == 0, true_fn=lambda: all_indices,
                          false_fn=lambda: mixing_weight_mask)
        indices = tf.cast(indices, dtype=tf.int32)
        intersection = tf.sets.intersection(tf.expand_dims(all_indices, axis=0), tf.expand_dims(indices, axis=0))
        intersection_values = tf.squeeze(tf.sparse.to_dense(intersection))

        model_weights = tf.reshape(tf.gather(self.model_weight, intersection_values), [-1])
        mixing_weight = tf.nn.softmax(model_weights)
        return mixing_weight

    def get_config(self):
        return {'size_shape': self.model_weight.shape}


def combine_weights_deltas(weight_deltas, mixing_weight):
    assert len(weight_deltas) == len(mixing_weight), f"Weights delta lens {len(weight_deltas)} != mixing weights lens {len(mixing_weight)}"
    # print(len(weight_deltas), mixing_weight)
    # for wd in weight_deltas:
    #     print('\t', [len(w) for w in wd])
    new_grad = [0 for _ in range(len(weight_deltas[0]))]
    for i in range(len(mixing_weight)):
        for j in range(len(weight_deltas[i])):
            new_grad[j] += mixing_weight[i] * tf.identity(weight_deltas[i][j])

    return new_grad


def mixing_weight_to_masked(mixing_weight, mixing_weight_mask=None):
    if mixing_weight_mask is None or len(mixing_weight_mask) == 0:
        return mixing_weight
    new_mixing_weight = []
    for ind in mixing_weight_mask:
        new_mixing_weight.append(mixing_weight[ind])
    new_mixing_weight = np.array(new_mixing_weight)
    new_mixing_weight = new_mixing_weight / np.sum(new_mixing_weight)
    return new_mixing_weight


def calc_optimal_mixing_weights(p_losses, scale=2, factor=1e-6):
    y = copy.deepcopy(p_losses)
    y = -y
    y1 = copy.deepcopy(y)
    y = y / scale
    y = np.exp(y)
    y = y / (np.sum(y, keepdims=True) + factor)
    return y, y1


def mixing_weight_process(mixing_weight_record, y, mixing_weight_mask, scale=2, factor=1e-5):
    y = mixing_weight_to_masked(y, mixing_weight_mask)
    y = y / scale
    y = np.exp(y)
    y = y / (np.sum(y, keepdims=True) + factor)
    y = y * mixing_weight_record
    y = y / (np.sum(y, keepdims=True) + factor)
    return y


""" Untested
https://github.com/ShuangtongLi/SCooL/blob/main/models/resnet.py#L1674
class AttentionLayerModelUpdateMultilayerMask(tf.keras.Model):
    def __init__(self, model, rank, out_dim=(10, 5)):
        super(AttentionLayerModelUpdateMultilayerMask, self).__init__()

        out_dim1, out_dim2 = out_dim
        self.w_qs_group1 = []
        self.w_qs_group2 = []
        self.num_weights = 0
        self.rank = rank

        for _ in model.weights:
            self.w_qs_group1.append(tf.keras.layers.Dense(out_dim1))
            self.w_qs_group2.append(tf.keras.layers.Dense(out_dim2))

            self.num_weights += 1

        self.w_qs_layers1 = self.w_qs_group1
        self.w_qs_layers2 = self.w_qs_group2

    def call(self, inputs, training=None, mask=None):
        model_update_group, mixing_weight_mask = inputs[0], inputs[1]
        q_group, k_group, neighbour_num = self.preprocess(model_update_group, mixing_weight_mask)

        prod_sum = tf.zeros(neighbour_num)
        for i in range(self.num_weights):
            q = q_group[i]
            k = k_group[i]
            neighbour_num, num_filter, in_dim = q.shape
            q = tf.tanh(self.w_qs_layers2[i](tf.tanh(self.w_qs_layers1[i](q))))
            k = tf.tanh(self.w_qs_layers2[i](tf.tanh(self.w_qs_layers1[i](k))))

            q = tf.reshape(q, (neighbour_num, -1))
            k = tf.reshape(k, (-1, 1))

            prod = tf.matmul(q, k)
            prod = tf.reshape(prod, (-1)) / num_filter
            prod_sum += prod

        prod_sum = tf.reshape(prod_sum, (-1)) / self.num_weights
        mixing_weight = tf.nn.softmax(prod_sum, axis=0)
        mixing_weight = tf.reshape(mixing_weight, (-1))
        return mixing_weight

    def preprocess(self, model_update_group, mixing_weight_mask=None):
        model_update_group1, val_grad = self.normalize_model_update_group(model_update_group)
        if mixing_weight_mask is None:
            grad_group = model_update_group1
        else:
            grad_group = [model_update_group1[i] for i in mixing_weight_mask]

        neighbour_num = len(grad_group)
        new_grad_group = []
        for i in range(self.num_weights):
            tmp_grad_group = []
            for j in range(neighbour_num):
                model_param = grad_group[j][i].numpy()
                model_param = model_param.reshape(model_param.shape[0], -1)
                tmp_grad_group.append(model_param)
            tmp_grad_group = tf.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)

        key_tensor_group = []
        for i in range(self.num_weights):
            model_param = val_grad[i].numpy()
            model_param = model_param.reshape(model_param.shape[0], -1)
            key_tensor_group.append(model_param)

        return new_grad_group, key_tensor_group, neighbour_num

    def normalize_model_update_group(self, model_update_group):
        new_grad_group = copy.deepcopy(model_update_group)
        new_grad_group = [tf.reshape(i, shape=(-1,)) for i in new_grad_group]
        new_grad_group = [i / (tf.norm(i) + 1e-10) for i in new_grad_group]
        new_grad_group = [tf.reshape(i, shape=model_update_group[0].shape) for i in new_grad_group]
        val_grad = new_grad_group[self.rank]
        return new_grad_group, val_grad

    def get_config(self):
        return {"num_weights", self.num_weights}
"""
