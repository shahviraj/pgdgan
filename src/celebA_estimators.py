# Files of this project is modified versions of 'https://github.com/AshishBora/csgm', which
#comes with the MIT licence: https://github.com/AshishBora/csgm/blob/master/LICENSE

import copy
import heapq
import tensorflow as tf
import numpy as np
import utils
import scipy.fftpack as fftpack
import pywt

import celebA_model_def
from celebA_utils import save_image


def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm='ortho').T, norm='ortho')


def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm='ortho').T, norm='ortho')


def vec(channels):
    image = np.zeros((64, 64, 3))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])


def devec(vector):
    image = np.reshape(vector, [64, 64, 3])
    channels = [image[:, :, i] for i in range(3)]
    return channels


def lasso_dct_estimator(hparams):  #pylint: disable = W0613
    """LASSO with DCT"""
    def estimator(A_val, y_batch_val, hparams):
        # One can prove that taking 2D DCT of each row of A,
        # then solving usual LASSO, and finally taking 2D ICT gives the correct answer.
        A_new = copy.deepcopy(A_val)
        for i in range(A_val.shape[1]):
            A_new[:, i] = vec([dct2(channel) for channel in devec(A_new[:, i])])

        x_hat_batch = []
        for j in range(hparams.batch_size):
            y_val = y_batch_val[j]
            z_hat = utils.solve_lasso(A_new, y_val, hparams)
            x_hat = vec([idct2(channel) for channel in devec(z_hat)]).T
            x_hat = np.maximum(np.minimum(x_hat, 1), -1)
            x_hat_batch.append(x_hat)
        return x_hat_batch
    return estimator


def dcgan_estimator(hparams):
    # pylint: disable = C0326

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    #A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.n_input), name='y_batch')

    # Create the generator
    z_batch = tf.Variable(tf.random_normal([hparams.batch_size, 100]), name='z_batch')
    x_hat_batch, restore_dict_gen, restore_path_gen = celebA_model_def.dcgan_gen(z_batch, hparams)

    # Create the discriminator
    prob, restore_dict_discrim, restore_path_discrim = celebA_model_def.dcgan_discrim(x_hat_batch, hparams)

    # measure the estimate

    y_hat_batch = tf.identity(x_hat_batch, name='y2_batch')


    # define all losses
    m_loss1_batch =  tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch =  tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
    zp_loss_batch =  tf.reduce_sum(z_batch**2, 1)
    d_loss1_batch = -tf.log(prob)
    d_loss2_batch =  tf.log(1-prob)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch \
                     + hparams.dloss1_weight * d_loss1_batch \
                     + hparams.dloss2_weight * d_loss2_batch
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)
    d_loss1 = tf.reduce_mean(d_loss1_batch)
    d_loss2 = tf.reduce_mean(d_loss2_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
    restorer_discrim = tf.train.Saver(var_list=restore_dict_discrim)
    restorer_gen.restore(sess, restore_path_gen)
    restorer_discrim.restore(sess, restore_path_discrim)

    def estimator(y_batch_val,z_batch_val,hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        assign_z_opt_op = z_batch.assign(z_batch_val)

        feed_dict = {y_batch: y_batch_val}


        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            sess.run(assign_z_opt_op)
            for j in range(hparams.max_update_iter):

                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val, \
                d_loss1_val, \
                d_loss2_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss,
                                        d_loss1,
                                        d_loss2], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {} d_loss1 {} d_loss2 {}'
                print logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val,
                                            d_loss1_val,
                                            d_loss2_val)

            x_hat_batch_val,z_batch_val,total_loss_batch_val = sess.run([x_hat_batch,z_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val,z_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator