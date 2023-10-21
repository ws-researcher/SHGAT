import os
import random
import numpy as np
import tensorflow as tf

# random.seed(0)  # 为python设置随机种子
# np.random.seed(0)  # 为numpy设置随机种子
# tf.random.set_seed(0)


from tensorflow.python.keras.backend import gather, relu

from tqdm import trange

from ADEA import ADEA

from DataProcess import load_data
from Util import l1, eval_alignment_by_sim_mat, eval_alignment_by_sim_mat_withName

def set_seeds(seed=0):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    # os.environ['TF_DETERMINISTIC_OPS'] = str(seed)

set_seeds(2)

def train(args):
    dataPath = args.dataDir

    train_pair, dev_pair, test_pair, data, AVH = load_data(dataPath, train_ratio=0.3, dev_ratio=0.7, test_ratio=0)

    node_size = len(data.get("ent_data").get("id_ent"))
    rel_size = len(data.get("rel_data").get("id_rel")) * 2 - 1
    attr_size = len(data.get("attr_data").get("id_attr"))

    batch_size = node_size

    addself_matrix = data.get("all_matrix").get("addself_bi")
    index_matrix = data.get("all_matrix").get("index_bi")
    attr_matrix = data.get("attr_matrix")
    model = ADEA(args, node_size=node_size, rel_size=rel_size, attr_size=attr_size, index_matrix=index_matrix,
                 all_matix=addself_matrix,
                 attr_matrix=attr_matrix)

    num_steps = 1 # len(train_data)/batch_size
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [3300 * num_steps, 6600 * num_steps],
        [0.001, 0.0005, 0.0001]
    )
    # optimizer = RMSprop(learning_rate=args.lr)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_fn)

    rest_set_1 = [e1 for e1, e2 in dev_pair]
    rest_set_2 = [e2 for e1, e2 in dev_pair]
    np.random.shuffle(rest_set_1)
    np.random.shuffle(rest_set_2)

    for turn in range(7):
        for i in trange(args.epoach):
            with tf.GradientTape() as tape:

                train_set = get_train_set(node_size, train_pair)

                ent_emb, conc_emb = model(AVH, training=True, epoach=turn*args.epoach + i)

                loss = align_loss(ent_emb, train_set) / batch_size

                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if i % 100 == 0:
                ent_emb, _ = model(AVH, training=False)
                CSLS_test(ent_emb, dev_pair)

        # new_pair = []
        # vec, _ = model(AVH, training=False)
        # Lvec = np.array([vec[e] for e in rest_set_1])
        # Rvec = np.array([vec[e] for e in rest_set_2])
        # Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        # Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        # A, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
        # B, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
        # A = sorted(list(A))
        # B = sorted(list(B))
        # for a, b in A:
        #     if B[b][1] == a:
        #         new_pair.append([rest_set_1[a], rest_set_2[b]])
        # print("generate new semi-pairs: %d." % len(new_pair))
        #
        # train_pair = np.concatenate([train_pair, np.array(new_pair)], axis=0)
        # for e1, e2 in new_pair:
        #     if e1 in rest_set_1:
        #         rest_set_1.remove(e1)
        #
        # for e1, e2 in new_pair:
        #     if e2 in rest_set_2:
        #         rest_set_2.remove(e2)


def get_train_set(node_size, train_pair):
    negative_ratio = node_size // len(train_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_pair, axis=0), axis=0, repeats=negative_ratio),
                           newshape=(-1, 2))
    np.random.shuffle(train_set)
    train_set = train_set[:node_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set


def align_loss(feature, train_set):
    samples_feature = gather(reference=feature, indices=train_set)
    h, t, nh, nt = [samples_feature[:, 0, :], samples_feature[:, 1, :], samples_feature[:, 2, :],
                    samples_feature[:, 3, :]]
    loss = relu(3 + l1(h, t) - l1(h, nt)) + relu(3 + l1(h, t) - l1(nh, t))
    loss = tf.reduce_sum(loss, keepdims=True)
    return loss


def CSLS_test(ent_emb, test_pair, thread_num=16, csls=10, accurate=True):
    Lvec = np.array([ent_emb[e1] for e1, e2 in test_pair])
    Rvec = np.array([ent_emb[e2] for e1, e2 in test_pair])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)

    eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_num, csls=csls, accurate=accurate)


def CSLS_test_withName(ent_emb, edict, test_pair, thread_num=16, csls=10, accurate=True):
    Lvec = np.array([ent_emb[e1] for e1, e2 in test_pair])
    Rvec = np.array([ent_emb[e2] for e1, e2 in test_pair])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)

    eName1 = np.array([edict[e1] for e1, e2 in test_pair])
    eName2 = np.array([edict[e2] for e1, e2 in test_pair])

    eval_alignment_by_sim_mat_withName(Lvec, Rvec, eName1, eName2, [1, 5, 10], thread_num, csls=csls, accurate=accurate)
