import os
import random
import numpy as np
import tensorflow as tf

# random.seed(0)  # 为python设置随机种子
# np.random.seed(0)  # 为numpy设置随机种子
# tf.random.set_seed(0)
# from tensorflow import keras
from tensorflow.keras import layers, initializers, activations
from tensorflow.python import keras
from tensorflow.python.keras.backend import gather, concatenate, dot, expand_dims

from tensorflow.python.keras.layers.core import Dropout


from Util import ra_drop
from moduleUtil import AVlyaer, MutilHeadAttention, MutilHeadAttention1, AVlyaer1

class ADEA(keras.Model):

    def __init__(self, arg, node_size, rel_size, attr_size, index_matrix=None, all_matix=None, attr_matrix=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = arg
        self.node_size = node_size
        self.rel_size = rel_size
        self.attr_size = attr_size
        self.index_matrix, self.all_matix, self.attr_matrix = np.array(index_matrix), np.array(all_matix), np.array(attr_matrix)

        self.ent_emb_layer = TokenEmbedding(node_size, self.args.input_dim, trainable=True)
        self.rel_emb_layer = TokenEmbedding(rel_size, self.args.rel_dim, trainable=True)
        self.attr_emb_layer = TokenEmbedding(attr_size, self.args.attr_dim, trainable=True)

        self.ent_emb = self.ent_emb_layer(1)
        self.rel_emb = self.rel_emb_layer(1)
        self.attr_emb = self.attr_emb_layer(1)

        self.wr = tf.keras.layers.Dense(1)
        self.wa = tf.keras.layers.Dense(1)

        self.entLayer = GRAT(arg)

        self.tanh = activations.get('tanh')
        self.relu = activations.get('relu')
        self.LeakyReLU = keras.layers.LeakyReLU(alpha=0.3)

    def call(self, AVH=None, training=None, mask=None, epoach=None):
        # -----------------------------------------------------------------------------------
        # litral_emb = self.litral_emb_layer(AVH)
        # -----------------------------------------------------------------------------------

        ent_rel_index = self.all_matix[:, 3:5].astype(np.int64)

        if training:
            ent_rel_index = ra_drop(ent_rel_index, dr=0.01)
        ent_rel_index = np.array(sorted(ent_rel_index, key=lambda x: (x[0], x[1])))
        rel_index, rel_id = tf.raw_ops.UniqueV2(x=ent_rel_index, axis=[0])

        ent_feature = gather(self.ent_emb, rel_index[:, 0])
        rel_feature = gather(self.rel_emb, rel_index[:, 1])
        feature = concatenate([ent_feature, rel_feature])
        attn = self.wr(feature)
        attn = self.LeakyReLU(attn)
        attn = tf.reshape(attn, (-1,))

        rel_adj = tf.SparseTensor(indices=rel_index, values=attn,
                                  dense_shape=(self.node_size, self.rel_size))
        rel_adj = tf.sparse.softmax(rel_adj)
        concept_rel = tf.sparse.sparse_dense_matmul(rel_adj, self.rel_emb)
        concept_rel = self.relu(concept_rel)

        # ent_rel_index = self.all_matix[:, 3:5].astype(np.int64)
        # ent_rel_index = np.array(sorted(ent_rel_index, key=lambda x: (x[0], x[1])))
        # rel_index, rel_id = tf.raw_ops.UniqueV2(x=ent_rel_index, axis=[0])
        # rel_adj = tf.SparseTensor(indices=rel_index, values=tf.ones_like(rel_index[:, 0], dtype="float32"),
        #                           dense_shape=(self.node_size, self.rel_size))
        # rel_adj = tf.sparse.to_dense(rel_adj)
        #
        # max_len = 0
        # for i in range(rel_adj.shape[0]):
        #     a = rel_adj[i]
        #     rel_ids = tf.reshape(tf.where(tf.equal(a, 1)), (-1,)).numpy()
        #     max_len = np.max((max_len, len(rel_ids)))
        #
        # ERH = []
        # for i in range(rel_adj.shape[0]):
        #     a = rel_adj[i]
        #     rel_ids = tf.reshape(tf.where(tf.equal(a, 1)), (-1,)).numpy()
        #     rel_feat = tf.gather(self.rel_emb, rel_ids)
        #     rel_num = rel_feat.shape[0]
        #     if rel_num != max_len:
        #         fill = tf.zeros((max_len - rel_num, self.args.rel_dim))
        #         rel_feat = concatenate([rel_feat, fill], axis=0)
        #     ERH.append(tf.expand_dims(rel_feat, axis=0))
        #
        # ERH = concatenate(ERH, axis=0)
        # concept_rel = self.relcept_emb_layer(ERH)

        # ------------------------------------------------------------------------------------
        ent_attr_index = self.attr_matrix[:, 0:2].astype(np.int64)
        if training:
            ent_attr_index = ra_drop(ent_attr_index, dr=0.01)
        ent_attr_index = np.array(sorted(ent_attr_index, key=lambda x: (x[0], x[1])))
        attr_index, _ = tf.raw_ops.UniqueV2(x=ent_attr_index, axis=[0])

        ent_feature = gather(self.ent_emb, attr_index[:, 0])
        attr_feature = gather(self.attr_emb, attr_index[:, 1])
        feature = concatenate([ent_feature, attr_feature])
        attn = self.wa(feature)
        attn = self.LeakyReLU(attn)
        attn = tf.reshape(attn, (-1,))

        attr_adj = tf.SparseTensor(indices=attr_index, values=attn,
                                   dense_shape=(self.node_size, self.attr_size))
        attr_adj = tf.sparse.softmax(attr_adj)
        concept_attr = tf.sparse.sparse_dense_matmul(attr_adj, self.attr_emb)
        concept_attr = self.relu(concept_attr)

        # ent_attr_index = self.attr_matrix[:, 0:2].astype(np.int64)
        # ent_attr_index = np.array(sorted(ent_attr_index, key=lambda x: (x[0], x[1])))
        # attr_index, _ = tf.raw_ops.UniqueV2(x=ent_attr_index, axis=[0])
        # attr_adj = tf.SparseTensor(indices=attr_index, values= tf.ones_like(attr_index[:, 0], dtype="float32"),
        #                            dense_shape=(self.node_size, self.attr_size))
        # attr_adj = tf.sparse.to_dense(attr_adj)
        #
        # max_len = 0
        # for i in range(attr_adj.shape[0]):
        #     a = attr_adj[i]
        #     attr_ids = tf.reshape(tf.where(tf.equal(a, 1)), (-1,)).numpy()
        #     max_len = np.max((max_len, len(attr_ids)))
        #
        # EAH = []
        # for i in range(attr_adj.shape[0]):
        #     a = attr_adj[i]
        #     attr_ids = tf.reshape(tf.where(tf.equal(a, 1)), (-1,)).numpy()
        #     attr_feat = tf.gather(self.attr_emb, attr_ids)
        #     attr_num = attr_feat.shape[0]
        #     if attr_num != max_len:
        #         fill = tf.zeros((max_len - attr_num, self.args.rel_dim))
        #         attr_feat = concatenate([attr_feat, fill], axis=0)
        #     EAH.append(tf.expand_dims(attr_feat, axis=0))
        #
        # EAH = concatenate(EAH, axis=0)
        #
        # concept_attr = self.attrcept_emb_layer(EAH)

        # ------------------------------------------------------------------------------------
        ent_emb = self.entLayer(self.ent_emb, rel_emb=self.rel_emb, attr_emb=self.attr_emb, index_matrix=self.index_matrix,
                                all_matix=self.all_matix, attr_matrix=self.attr_matrix, concept_attr=concept_attr,
                                concept_rel=concept_rel, training=training, epoach=epoach)

        # ------------------------------------------------------------------------------------
        ent_emb = concatenate([ent_emb, concept_rel, concept_attr])
        # ent_emb = concatenate([ent_emb, concept_rel])
        # ent_emb = concatenate([ent_emb, litral_emb, concept_rel, concept_attr])

        conc_emb = concatenate([concept_rel, concept_attr])
        if training:
            ent_emb = Dropout(self.args.dropout_rate)(ent_emb)
            return ent_emb, conc_emb
        else:
            return ent_emb, conc_emb


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        # self.embeddings_initializer = initializers.get("glorot_uniform")

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings

class GRAT(layers.Layer):
    def __init__(self, arg, **kwargs):
        super().__init__(**kwargs)
        self.args = arg

        self.ent_attn_kernels = []
        self.rel_attn_kernels = []

        self.ent_kernels = []
        self.rel_kernels = []

        self.concept_rel_attn_kernels = []
        self.concept_rel_kernels = []

        self.concept_attr_attn_kernels = []
        self.concept_attr_kernels = []

        self.LeakyReLU = keras.layers.LeakyReLU(alpha=0.3)
        self.tanh = activations.get('tanh')
        self.relu = activations.get('relu')

    def build(self, input_shape):
        for l in range(self.args.num_layers):

            self.ent_attn_kernels.append([])
            self.rel_attn_kernels.append([])

            self.ent_kernels.append([])
            self.rel_kernels.append([])

            self.concept_rel_attn_kernels.append([])
            self.concept_rel_kernels.append([])

            self.concept_attr_attn_kernels.append([])
            self.concept_attr_kernels.append([])

            for head in range(self.args.head):
                self.ent_attn_kernels[l].append(tf.keras.layers.Dense(1))
                self.concept_attr_attn_kernels[l].append(tf.keras.layers.Dense(1))

    def call(self, inputs, **kwargs):
        ent_emb = inputs
        rel_emb = kwargs.get("rel_emb")

        concept_attr = kwargs.get("concept_attr")
        concept_rel = kwargs.get("concept_rel")
        node_size = ent_emb.shape[0]

        all_matix = kwargs.get("all_matix")

        ent_rel_index = all_matix[:, 3:5].astype(np.int64)

        ent_rel_index = np.array(sorted(ent_rel_index, key=lambda x: (x[0], x[1])))

        ent_ent_index = all_matix[:, 0:2].astype(np.int64)

        ent_ent_index = np.array(sorted(ent_ent_index, key=lambda x: (x[0], x[1])))
        index, idx = tf.raw_ops.UniqueV2(x=ent_ent_index, axis=[0])


        ent_adj = tf.SparseTensor(indices=index, values=tf.ones_like(index[:, 0], dtype="float32"),
                                  dense_shape=(node_size, node_size))
        ent_adj = tf.sparse.softmax(ent_adj)
        ent_emb = tf.sparse.sparse_dense_matmul(ent_adj, ent_emb)

        outputs = []
        for l in range(self.args.num_layers):
            ent_emb = self.relu(ent_emb)
            head_emb_list = tf.transpose(tf.reshape(ent_emb, (node_size, self.args.head, -1)), perm=[1, 0, 2])
            head_feature_list = []
            for head in range(self.args.head):
                ent_emb = head_emb_list[head]

                ent_attn_kernel = self.ent_attn_kernels[l][head]
                concept_attn_kernel = self.concept_attr_attn_kernels[l][head]

                neighs_concept_rel_feature = gather(concept_rel, index[:, 1])
                neighs_concept_attr_feature = gather(concept_attr, index[:, 1])
                neighs_concept_feature = concatenate([neighs_concept_rel_feature, neighs_concept_attr_feature])

                self_concept_rel_feature = gather(concept_rel, index[:, 0])
                self_concept_attr_feature = gather(concept_attr, index[:, 0])
                self_concept_feature = concatenate([self_concept_rel_feature, self_concept_attr_feature])

                neighs_feature = gather(ent_emb, index[:, 1])
                self_feature = gather(ent_emb, index[:, 0])
                rels_feature = gather(rel_emb, ent_rel_index[:, 1])
                rels_feature = tf.math.segment_mean(rels_feature, idx)


                ent_attn = ent_attn_kernel(concatenate([self_feature, rels_feature, neighs_feature]))
                ent_attn = tf.reshape(ent_attn, (-1,))
                ent_attn = self.LeakyReLU(ent_attn)

                concept_attn = concept_attn_kernel(concatenate([self_concept_feature, rels_feature, neighs_concept_feature]))
                concept_attn = tf.reshape(concept_attn, (-1,))
                concept_attn = self.LeakyReLU(concept_attn)

                attn = ent_attn * concept_attn
                attn = tf.nn.softmax(attn, axis=-1)
                attn = tf.SparseTensor(indices=index, values=attn, dense_shape=(node_size, node_size))
                attn = tf.sparse.softmax(attn)

                new_ent_emb = tf.sparse.sparse_dense_matmul(attn, ent_emb)

                head_feature_list.append(new_ent_emb)

            ent_feature = concatenate(head_feature_list)
            ent_feature = self.tanh(ent_feature)

            ent_emb = ent_feature
            outputs.append(ent_feature)

        outputs = concatenate(outputs)
        return outputs



# class GRAT(layers.Layer):
#     def __init__(self, arg, **kwargs):
#         super().__init__(**kwargs)
#         self.args = arg
#
#         self.ent_attn_kernels = []
#         self.rel_attn_kernels = []
#
#         self.ent_kernels = []
#         self.rel_kernels = []
#
#         self.concept_rel_attn_kernels = []
#         self.concept_rel_kernels = []
#
#         self.concept_attr_attn_kernels = []
#         self.concept_attr_kernels = []
#
#         self.LeakyReLU = keras.layers.LeakyReLU(alpha=0.3)
#         self.tanh = activations.get('tanh')
#         self.relu = activations.get('relu')
#
#     def build(self, input_shape):
#         for l in range(self.args.num_layers):
#
#             self.ent_attn_kernels.append([])
#             self.rel_attn_kernels.append([])
#
#             self.ent_kernels.append([])
#             self.rel_kernels.append([])
#
#             self.concept_rel_attn_kernels.append([])
#             self.concept_rel_kernels.append([])
#
#             self.concept_attr_attn_kernels.append([])
#             self.concept_attr_kernels.append([])
#
#             num_head = self.args.head
#             output_dim_head = int(self.args.output_dim/num_head)
#             rel_dim_head = self.args.rel_dim
#             attr_dim_head = self.args.attr_dim
#
#             for head in range(num_head):
#                 self.ent_attn_kernels[l].append(tf.keras.layers.Dense(1))
#                 # self.ent_kernels[l].append(tf.keras.layers.Dense(output_dim_head))
#                 # self.concept_rel_kernels[l].append(tf.keras.layers.Dense(2*rel_dim_head))
#                 self.concept_attr_attn_kernels[l].append(tf.keras.layers.Dense(1))
#                 # self.concept_attr_kernels[l].append(tf.keras.layers.Dense(2*attr_dim_head))
#
#     def call(self, inputs, **kwargs):
#         ent_emb = inputs
#         concept_attr = kwargs.get("concept_attr")
#         concept_rel = kwargs.get("concept_rel")
#         node_size = ent_emb.shape[0]
#
#         all_matix = kwargs.get("all_matix")
#         ent_ent_index = all_matix[:, 0:2].astype(np.int64)
#         ent_ent_index = np.array(sorted(ent_ent_index, key=lambda x: (x[0], x[1])))
#         index, idx = tf.raw_ops.UniqueV2(x=ent_ent_index, axis=[0])
#
#         ent_adj = tf.SparseTensor(indices=index, values=tf.ones_like(index[:, 0], dtype="float32"),
#                                   dense_shape=(node_size, node_size))
#         ent_adj = tf.sparse.softmax(ent_adj)
#         ent_emb = tf.sparse.sparse_dense_matmul(ent_adj, ent_emb)
#
#         outputs = []
#         for l in range(self.args.num_layers):
#             ent_emb = self.relu(ent_emb)
#             head_emb_list = tf.transpose(tf.reshape(ent_emb, (node_size, self.args.head, -1)), perm=[1, 0, 2])
#             head_feature_list = []
#             for head in range(self.args.head):
#                 ent_emb = head_emb_list[head]
#
#                 ent_attn_kernel = self.ent_attn_kernels[l][head]
#                 concept_attn_kernel = self.concept_attr_attn_kernels[l][head]
#
#                 # ent_kernel = self.ent_kernels[l][head]
#                 # concept_rel_kernel = self.concept_rel_kernels[l][head]
#                 # concept_attr_kernel = self.concept_attr_kernels[l][head]
#
#                 neighs_concept_rel_feature = gather(concept_rel, index[:, 1])
#                 neighs_concept_attr_feature = gather(concept_attr, index[:, 1])
#                 neighs_concept_feature = concatenate([neighs_concept_rel_feature, neighs_concept_attr_feature])
#                 # neighs_concept_feature = concept_rel_kernel(neighs_concept_feature)
#                 # neighs_concept_feature = self.relu(neighs_concept_feature)
#
#                 self_concept_rel_feature = gather(concept_rel, index[:, 0])
#                 self_concept_attr_feature = gather(concept_attr, index[:, 0])
#                 self_concept_feature = concatenate([self_concept_rel_feature, self_concept_attr_feature])
#                 # self_concept_feature = concept_attr_kernel(self_concept_feature)
#                 # self_concept_feature = self.relu(self_concept_feature)
#
#                 neighs_feature = gather(ent_emb, index[:, 1])
#                 self_feature = gather(ent_emb, index[:, 0])
#
#                 # neighs_feature = ent_kernel(neighs_feature)
#                 # neighs_feature = self.relu(neighs_feature)
#                 # self_feature = ent_kernel(self_feature)
#                 # self_feature = self.relu(self_feature)
#
#                 ent_attn = ent_attn_kernel(concatenate([self_feature, neighs_feature]))
#                 ent_attn = tf.reshape(ent_attn, (-1,))
#                 ent_attn = self.LeakyReLU(ent_attn)
#
#                 concept_attn = concept_attn_kernel(concatenate([self_concept_feature, neighs_concept_feature]))
#                 concept_attn = tf.reshape(concept_attn, (-1,))
#                 concept_attn = self.LeakyReLU(concept_attn)
#
#                 attn = ent_attn * concept_attn
#                 attn = tf.nn.softmax(attn, axis=-1)
#                 attn = tf.SparseTensor(indices=index, values=attn, dense_shape=(node_size, node_size))
#                 attn = tf.sparse.softmax(attn)
#
#                 # new_ent_emb = tf.math.segment_sum(neighs_feature * expand_dims(attn.values, axis=-1), index[:, 0])
#                 new_ent_emb = tf.sparse.sparse_dense_matmul(attn, ent_emb)
#
#                 head_feature_list.append(new_ent_emb)
#
#             ent_feature = concatenate(head_feature_list)
#             ent_feature = self.tanh(ent_feature)
#
#             ent_emb = ent_feature
#             outputs.append(ent_feature)
#
#         outputs = concatenate(outputs)
#         return outputs

