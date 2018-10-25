# encoding:UTF-8
import numpy as np
from keras.layers import Layer
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import pickle
from keras import models, optimizers, regularizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap
import glob

K.set_image_data_format('channels_last')
import tensorflow as tf
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

# activation functions
activation = 'relu'
last_activation = 'linear'


# eye model
def get_eye_model(img_cols, img_rows, img_ch):

    eye_img_input = Input(shape=(img_cols, img_rows, img_ch))

    """
    h = Conv2D(96, (3, 3), activation=activation)(eye_img_input)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(256, (3, 3), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(384, (3, 3), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    out = Conv2D(64, (1, 1), activation=activation)(h)
    """

    h = Conv2D(96, (5, 5), activation=activation)(eye_img_input)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(256, (5, 5), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(384, (3, 3), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    out = Conv2D(64, (1, 1), activation=activation)(h)
    model = Model(inputs=eye_img_input, outputs=out)

    return model


# final model
def get_eye_tracker_model(img_cols, img_rows, img_ch):

    # get partial models
    eye_net = get_eye_model(img_cols, img_rows, img_ch)

    # right eye model
    eye_input = Input(shape=(img_cols, img_rows, img_ch))
    e = eye_net(eye_input)

    # dense layers for eyes
    e = Flatten()(e)
    e = BatchNormalization()(e)
    fc1 = Dense(128, activation=activation)(e)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dense(128, activation=activation)(fc1)
    fc1 = BatchNormalization()(fc1)
    fc2 = Dense(64, activation=activation)(fc1)
    fc2 = BatchNormalization()(fc2)
    fc3 = Dense(2, activation=last_activation)(fc2)

    # final model
    final_model = Model(
        inputs=[eye_input],
        outputs=[fc3])

    return final_model

def batch_iter(mode, base_batch_dir):
    batch_dir = join(base_batch_dir, mode, 'batches')
    batch_list = os.listdir(batch_dir)
    num_batches = len(batch_list)
    def train_generator():
        batch_path_list = glob.glob(batch_dir + '/*')
        while True:
            for batch_path in batch_path_list:
                with open(batch_path, 'rb') as f:
                    batch = pickle.load(f)
                yield batch[0], batch[1]
    return num_batches, train_generator()

def train(model, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_loss',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=['mae']
                  )

    """
    # Training without data augmentation:
    model.fit(x_train, [y_train, eye_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[x_test, [y_test, eye_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """
# augmentなし
    # Begin: Training with data augmentation ---------------------------------------------------------------------#

    num_train_batches, train_generator = batch_iter('train', args.batch_dir)
    num_val_batches, val_generator = batch_iter('val', args.batch_dir)
    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=num_train_batches,
                        epochs=args.epochs,
                        validation_data=val_generator,
                        validation_steps=num_val_batches,
                        callbacks=[log, tb, checkpoint, lr_decay, early_stopping])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model
if __name__ == "__main__":
    import os
    from os.path import join
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--batch_dir', default='/home/docker/share/eye_tracking/data/')
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--n_class', default=8, type=int)
    parser.add_argument('--dim_capsule', default=8, type=int)
    parser.add_argument('--lr', default=0.01, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=2.0, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = get_eye_tracker_model(32, 32, 3)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        eval_model.load_weights(args.weights)
    if not args.testing:
        train(model=model, args=args)
    else:
        if args.weights is None:
            print('No weight')
        test(model=eval_model, args=args)
