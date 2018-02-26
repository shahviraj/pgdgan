# Files of this project is modified versions of 'https://github.com/AshishBora/csgm', which
#comes with the MIT licence: https://github.com/AshishBora/csgm/blob/master/LICENSE

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils

def main(hparams):
    # Set up some stuff according to hparams
    hparams.n_input = np.prod(hparams.image_shape)
    maxiter = hparams.max_outer_iter
    utils.print_hparams(hparams)

    # get inputs
    xs_dict = model_input(hparams)

    estimators = utils.get_estimators(hparams)
    utils.setup_checkpointing(hparams)
    measurement_losses, l2_losses = utils.load_checkpoints(hparams)

    x_hats_dict = {'dcgan' : {}}
    x_batch_dict = {}
    for key, x in xs_dict.iteritems():
        if hparams.lazy:
            # If lazy, first check if the image has already been
            # saved before by *all* estimators. If yes, then skip this image.
            save_paths = utils.get_save_paths(hparams, key)
            is_saved = all([os.path.isfile(save_path) for save_path in save_paths.values()])
            if is_saved:
                continue

        x_batch_dict[key] = x
        if len(x_batch_dict) < hparams.batch_size:
            continue

        # Reshape input
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.iteritems()]
        x_batch = np.concatenate(x_batch_list)

        # Construct measurements
        A_outer = utils.get_outer_A(hparams)

        y_batch_outer=np.matmul(x_batch, A_outer)


        x_main_batch = 0.0 * x_batch
        z_opt_batch = np.random.randn(hparams.batch_size, 100)
        for k in range(maxiter):

            x_est_batch=x_main_batch + hparams.outer_learning_rate*(np.matmul((y_batch_outer-np.matmul(x_main_batch,A_outer)),A_outer.T))



            estimator = estimators['dcgan']
            x_hat_batch,z_opt_batch = estimator(x_est_batch,z_opt_batch, hparams)
            x_main_batch=x_hat_batch


        for i, key in enumerate(x_batch_dict.keys()):
            x = xs_dict[key]
            y = y_batch_outer[i]
            x_hat = x_hat_batch[i]

            # Save the estimate
            x_hats_dict['dcgan'][key] = x_hat

            # Compute and store measurement and l2 loss
            measurement_losses['dcgan'][key] = utils.get_measurement_loss(x_hat, A_outer, y)
            l2_losses['dcgan'][key] = utils.get_l2_loss(x_hat, x)
        print 'Processed upto image {0} / {1}'.format(key+1, len(xs_dict))

        # Checkpointing
        if (hparams.save_images) and ((key+1) % hparams.checkpoint_iter == 0):
            utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, save_image, hparams)
            #x_hats_dict = {'dcgan' : {}}
            print '\nProcessed and saved first ', key+1, 'images\n'

        x_batch_dict = {}

    # Final checkpoint
    if hparams.save_images:
        utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, save_image, hparams)
        print '\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict))

    if hparams.print_stats:
        for model_type in hparams.model_types:
            print model_type
            mean_m_loss = np.mean(measurement_losses[model_type].values())
            mean_l2_loss = np.mean(l2_losses[model_type].values())
            print 'mean measurement loss = {0}'.format(mean_m_loss)
            print 'mean l2 loss = {0}'.format(mean_l2_loss)

    if hparams.image_matrix > 0:
        utils.image_matrix(xs_dict, x_hats_dict, view_image, hparams)

    # Warn the user that some things were not processsed
    if len(x_batch_dict) > 0:
        print '\nDid NOT process last {} images because they did not fill up the last batch.'.format(len(x_batch_dict))
        print 'Consider rerunning lazily with a smaller batch size.'


if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--pretrained-model-dir', type=str, default='./models/celebA_64_64/', help='Directory containing pretrained model')

    # Input
    PARSER.add_argument('--input-type', type=str, default='full-input', help='Where to take input from')
    PARSER.add_argument('--input-path-pattern', type=str, default='./data/celebAtest/*.jpg', help='Pattern to match to get images')
    PARSER.add_argument('--num-input-images', type=int, default=64, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=64, help='How many examples are processed together')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type: as of now supports only gaussian')

    # Measurement type specific hparams

    PARSER.add_argument('--num-outer-measurements', type=int, default=1000, help='number of gaussian measurements(outer)')

    # Model
    PARSER.add_argument('--model-types', type=str, nargs='+', default=['dcgan'], help='model(s) used for estimation')
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=1.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior_weight', type=float, default=0.001, help='weight on z prior')
    PARSER.add_argument('--dloss1_weight', type=float, default=0.0, help='-log(D(G(z))')
    PARSER.add_argument('--dloss2_weight', type=float, default=0.0, help='log(1-D(G(z))')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='adam', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=100, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=10, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')
    PARSER.add_argument('--outer-learning-rate', type=float, default=0.5, help='learning rate of outer loop GD')
    PARSER.add_argument('--max-outer-iter', type=int, default=10, help='maximum no. of iterations for outer loop GD')


    # Output
    PARSER.add_argument('--lazy', action='store_true', help='whether the evaluation is lazy')
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--save-stats', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=50, help='checkpoint every x batches')
    PARSER.add_argument('--image-matrix', type=int, default=1,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                       )

    HPARAMS = PARSER.parse_args()


    HPARAMS.image_shape = (64, 64, 3)
    from celebA_input import model_input
    from celebA_utils import view_image, save_image


    main(HPARAMS)
