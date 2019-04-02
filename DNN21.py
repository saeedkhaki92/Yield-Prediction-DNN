import numpy as np

import tensorflow as tf

import matplotlib.pylab as plt



def Fc_layer(input, Cache, flag, channel_in, channel_out, Dropout, keepProb, is_training, name="FC"):
    with tf.name_scope(name):



        W1 = tf.get_variable(name=name + 'W1', shape=[channel_out, channel_in], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        W2 = tf.get_variable(name=name + 'W2', shape=[channel_out, channel_in], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        Z1 = tf.matmul(W1, input)
        Z1 = tf.transpose(Z1)
        param_initializers = {'beta': tf.zeros_initializer(), 'gamma': tf.ones_initializer()}

        Z1 = tf.contrib.layers.batch_norm(Z1, decay=0.99, center=True, scale=True,
                                          updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=is_training,
                                          data_format='NHWC', param_initializers=param_initializers,
                                          scope=name + '2b')
        Z1 = tf.transpose(Z1)

        Z2 = tf.matmul(W2, input)
        Z2 = tf.transpose(Z2)
        param_initializers = {'beta': tf.zeros_initializer(), 'gamma': tf.ones_initializer()}

        Z2 = tf.contrib.layers.batch_norm(Z2, decay=0.99, center=True, scale=True,
                                          updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=is_training,
                                          data_format='NHWC', param_initializers=param_initializers,
                                          scope=name + '2a')

        Z2 = tf.transpose(Z2)


        if flag == False:

            A = tf.maximum(Z1, Z2)

        else:

            A = tf.maximum(Z1 + Cache, Z2 + Cache)


        if Dropout:
            A = tf.nn.dropout(A, keep_prob=keepProb, noise_shape=[channel_out, 1])
            print("Dropout_added", keepProb)

        L2 = tf.norm(W1, ord=2) + tf.norm(W2, ord=2)

    return A, L2



def Fc_layerL(input, channel_in, channel_out, Dropout, keep_prob, name="FCL"):
    with tf.name_scope(name):

        W1 = tf.get_variable(name=name + 'W1', shape=[channel_out, channel_in], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable(name=name + 'b1', shape=[channel_out, 1], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

        Z1 = tf.matmul(W1, input) + b1

        A = tf.maximum(Z1, 0.0)

        if Dropout:
            A = tf.nn.dropout(A, keep_prob=keep_prob, noise_shape=[channel_out, 1])
            print("Added Dropout", keep_prob)

        L1 = tf.norm(W1, ord=1)
    return A, L1


def Fc_layerLL(input, Cache, channel_in, channel_out, Dropout, keep_prob, is_training, name="FCL"):
    with tf.name_scope(name):


        W1 = tf.get_variable(name=name + 'W1', shape=[channel_out, channel_in], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable(name=name + 'b1', shape=[channel_out, 1], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

        Z = tf.matmul(W1, input) + b1

        A = Z

        if Dropout:
            A = tf.nn.dropout(A, keep_prob=1, noise_shape=[channel_out, 1])

        L2 = tf.norm(W1, ord=2)

    return A, L2


def Cost_function(Y, Yhat):
    E = Y - Yhat
    E2 = tf.pow(E, 2)

    MSE = tf.squeeze(tf.reduce_mean(E2, axis=1))
    RMSE = tf.pow(MSE, 0.5)
    Loss = tf.losses.huber_loss(Y, Yhat, weights=1.0, delta=1.0)
    return RMSE, MSE, E, Loss


def main_model(X_training,Y_training,X_test, Y_test, layer, Max_it,learning_rate,batch_size_tr, batch_size_te,keepProb,gamma1, gamma2, Dropout=False):




    with tf.device("/cpu:0"):

        X_t = tf.placeholder(shape=[n_x, None], dtype=tf.float32)
        Y_t = tf.placeholder(shape=[1, None], dtype=tf.float32)
        is_tarining = tf.placeholder(dtype=tf.bool)
        lr = tf.placeholder(dtype=tf.float32)


        H, L1 = Fc_layerL(X_t, n_x, layer[1], Dropout, keepProb[1], name="FC1")


        Cashe = H

        L2 = 0
        for i in range(1, len(layer) - 2):

            Name = "FCC" + str(i + 1)



            if i % 2 == 0:
                flag = True

                H, L = Fc_layer(H, Cashe, flag, layer[i], layer[i + 1], Dropout, keepProb[i + 1], is_tarining,
                                name=Name)


                Cashe = H

                print("Shortcut Added", i + 1)


            else:
                flag = False
                H, L = Fc_layer(H, Cashe, flag, layer[i], layer[i + 1], Dropout, keepProb[i + 1], is_tarining,
                                name=Name)

            L2 = L2 + L

        Dropout1 = False
        # H = Cashe + H

        Yhat3, L = Fc_layerLL(H, Cashe, layer[-2], layer[-1], Dropout1, 1, is_tarining, name="FC_L")

        L2 = L2 + L

        print(L2)
        print(Yhat3)
        Yhat3 = tf.identity(Yhat3, name='Yhat')

        with tf.name_scope("Cost_function3"):

            RMSE3, MSE, E, Loss3 = Cost_function(Y_t, Yhat3)

        print(RMSE3)

        RMSE3 = tf.identity(RMSE3, name='RMSE')

        TLoss = tf.constant(1.0, dtype=tf.float32) * Loss3 \
                + tf.constant(gamma1, tf.float32) * L1 + tf.constant(gamma2, tf.float32) * L2

        with tf.name_scope("Train"):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(lr).minimize(TLoss)

        init = tf.global_variables_initializer()
        sess = tf.Session()

        sess.run(init)

        m_t = X_training.shape[1]
        LossTr = []
        LossTe = []

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            print(variable)
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                #   print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        print("total_parameters", total_parameters)

        for i in range(Max_it):

            if i%50000==0 and i>0:

                learning_rate=learning_rate/2

                print('learning rate decayed',learning_rate)



            I = np.random.randint(m_t, size=batch_size_tr)

            Batch_X = X_training[:, I]
            Batch_Y = Y_training[0, I]
            Batch_Y = Batch_Y.reshape(1, len(Batch_Y))

            sess.run(train_op, feed_dict={X_t: Batch_X, Y_t: Batch_Y, is_tarining: True,lr:learning_rate})

            if i % 5000 == 0:

                I1 = np.random.randint(X_test.shape[1], size=batch_size_te)

                rmse_tr,yhat_tr_n = sess.run([RMSE3,Yhat3], feed_dict={X_t:X_training, Y_t: Y_training, is_tarining: False})

                ctr = np.corrcoef(np.squeeze(yhat_tr_n), np.squeeze(Y_training))[0, 1]

                rmse_te,yhat_te_n = sess.run([RMSE3,Yhat3],
                                   feed_dict={X_t: X_test[:, I1], Y_t: Y_test[0, I1].reshape(1, len(Y_test[0, I1])),
                                              is_tarining: False})

                LossTe.append(rmse_te)
                LossTr.append(rmse_tr)

                ce = np.corrcoef(np.squeeze(yhat_te_n), np.squeeze(Y_test[0, I1]))[0, 1]
                print("Iteration %d , The training RMSE is %f  and C is %f and test RMSE is %f  and C %f" % (i, rmse_tr, ctr,rmse_te,ce))


        rmse_te, Yhat_te = sess.run([RMSE3, Yhat3],
                                    feed_dict={X_t: X_test[:, 0:], Y_t: Y_test[:, 0:], is_tarining: False})
        rmse_tr, Yhat_tr = sess.run([RMSE3, Yhat3],
                                    feed_dict={X_t: X_training, Y_t: Y_training, is_tarining: False})

    saver = tf.train.Saver()
    saver.save(sess, './saved_model', global_step=i)
    ctr = np.corrcoef(np.squeeze(Yhat_tr), np.squeeze(Y_training))[0, 1]
    print(ctr)

    ce = np.corrcoef(np.squeeze(Yhat_te), np.squeeze(Y_test[:, 0:]))[0, 1]
    print('test C',ce)
    return rmse_tr, rmse_te, Yhat_tr, Yhat_te, Y_training, Y_test, LossTr, LossTe



X_training=np.load('X_training')
Y_training=np.load('Y_training')
X_test=np.load('X_test')
Y_test=np.load('Y_test')


n_x = X_training.shape[0]
n_l1 = 21  # Number of layers with n1 neurons
n_l2 = 0  #  Number of layers with n2 neurons
n_l3 = 0  #  Number of layers with n3 neurons

L1 = np.ones([1, n_l1], dtype=int) * 50 #Number of neurons n1
L2 = np.ones([1, n_l2], dtype=int) * 50 ##Number of neurons n2
L3 = np.ones([1, n_l3], dtype=int) * 50  # #Number of neurons n3
layer = np.concatenate(([[n_x]], L1, L2, L3, [[1]]), axis=1)
layer = np.squeeze(layer)
print(layer.shape)
Max_it = 300000  # Max iteration
learning_rate = 0.0003  # Learning rate




gamma1 = 0.006  # 0.006  # L1 regularization amount for Layer 1
gamma2 = 0.002  # 0.2   # L2 regularization amount for other layers

K1 = np.ones([1, n_l1], dtype=float) * 1
K2 = np.ones([1, n_l2], dtype=float) * 1
K3 = np.ones([1, n_l3], dtype=float) * 1

keepProb = np.concatenate(([[1]], K1, K2, K3, [[1]]), axis=1)

keepProb = np.squeeze(keepProb)
keepProb[-2] = 1
keepProb[-3] = 1
keepProb[2] = 1
print(keepProb)
batch_size_tr = 64  #train batch size
batch_size_te = 64 # test batch size

print("Training_shape", X_training.shape)
# print("Test_shape", X_test.shape)




rmse_tr, rmse_te, Yhat_tr, Yhat_te, Y_training, Y_test, LossTr, LossTe = main_model(X_training,Y_training,X_test, Y_test, layer, Max_it,
                                                                                    learning_rate,
                                                                                    batch_size_tr, batch_size_te,
                                                                                    keepProb,
                                                                                    gamma1, gamma2, Dropout=False)

print('***********', rmse_tr, rmse_te)

fig, ax = plt.plt.subplots(figsize=(10, 8))

I1 = np.random.randint(Yhat_tr.shape[1], size=100)

pred = Yhat_tr[0, I1]
GT = Y_training[0, I1]

# fig, ax = plt.subplots(figsize=(10,8))

# plot a black line between the
# ith prediction and the ith ground truth
for i in range(len(pred)):
    ax.plot([i, i], [pred[i], GT[i]], c="k", linewidth=0.8)
ax.plot(pred, 'o', label='Prediction', markersize=8, color='g')
ax.plot(GT, '^', label='Ground Truth', markersize=8, color='r')

ax.set_xlim((-1, 101))
plt.xlabel('Experimental hybrids')
plt.ylabel('Yield')
plt.title('Performance on Training Set')

plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
# plt.subplot(132)
I2 = np.random.randint(Yhat_te.shape[1], size=100)

pred = Yhat_te[0, I2]
GT = Y_test[0, I2]

# fig, ax = plt.subplots(figsize=(10,8))

# plot a black line between the
# ith prediction and the ith ground truth
for i in range(len(pred)):
    ax.plot([i, i], [pred[i], GT[i]], c="k", linewidth=0.8)
ax.plot(pred, 'o', label='Prediction', markersize=8, color='g')
ax.plot(GT, '^', label='Ground Truth', markersize=8, color='r')

ax.set_xlim((-1, 101))

plt.xlabel('Experimental hybrids')
plt.ylabel('Yield')
plt.title('Performance on Validation Set')

plt.legend()
plt.show()

plt.figure(1)

# plt.subplot(131)
I1 = np.random.randint(Yhat_tr.shape[1], size=100)
a1, = plt.plot(Yhat_tr[0, I1], 'r--', label="Prediction")

a2, = plt.plot(Y_training[0, I1], 'b--', label="Target")
plt.legend(handles=[a1, a2])

plt.title('Performance on Training Set')

plt.xlabel("Index")
plt.ylabel("Yield")
plt.show()

plt.figure(2)
# plt.subplot(132)
I2 = np.random.randint(Yhat_te.shape[1], size=100)
a1, = plt.plot(Yhat_te[0, I2], 'r--', label="Prediction")
a2, = plt.plot(Y_test[0, I2], 'b--', label="Target")
plt.legend(handles=[a1, a2])
plt.xlabel("Index")
plt.ylabel("Yield")
plt.title('Performance on  Validation set')
plt.show()

plt.figure(3)
# plt.subplot(133)

a1, = plt.plot(LossTe, 'r--', label="Test_Lost")
a2, = plt.plot(LossTr, 'b--', label="Train_Lost")
plt.legend(handles=[a1, a2])
plt.xlabel("Index")
plt.ylabel("Loss")
plt.title('Loss_function')

plt.show()

print(Yhat_te)