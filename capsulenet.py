# encoding:utf-8
"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
import pickle
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap
import glob

# メモリを先食いしないようにする呪文
K.set_image_data_format('channels_last')
import tensorflow as tf
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)



def CapsNet(input_shape, n_class, routings, dim_capsule=32):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv2')(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv2, dim_capsule=8, n_channels=16, kernel_size=3, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_capsule, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
# 回帰用に変更の必要あり
    #out_caps = Length(name='capsnet')(digitcaps)
    regression = models.Sequential(name='regression')
    regression.add(layers.Reshape(target_shape=(n_class*dim_capsule,), input_shape=(n_class, dim_capsule)))
    regression.add(layers.Dense(256, activation='relu', input_dim=n_class*dim_capsule))
    regression.add(layers.Dense(512, activation='relu'))
    regression.add(layers.Dropout(0.5))
    regression.add(layers.Dense(2))
    out_caps = regression(digitcaps)

    # Decoder network.
# 入力削除、maskも不要
    #y = layers.Input(shape=(n_class,))
    #masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    #masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder_leye = models.Sequential(name='decoder_leye')
    decoder_leye.add(layers.Lambda(lambda x: x[:, 0], output_shape=(dim_capsule,), input_shape=(n_class, dim_capsule)))
    decoder_leye.add(layers.Dense(512, activation='relu', input_dim=dim_capsule))
    decoder_leye.add(layers.Dense(1024, activation='relu'))
    decoder_leye.add(layers.Dropout(0.5))
# input_shape -> eye_shape
# no activation(linear)
    decoder_leye.add(layers.Dense(32*32*1))

#output_shape=(16,), 
    decoder_reye = models.Sequential(name='decoder_reye')
    decoder_reye.add(layers.Lambda(lambda x: x[:, 1], output_shape=(dim_capsule,), input_shape=(n_class, dim_capsule)))
    decoder_reye.add(layers.Dense(512, activation='relu', input_dim=dim_capsule))
    decoder_reye.add(layers.Dense(1024, activation='relu'))
    decoder_leye.add(layers.Dropout(0.5))
# input_shape -> eye_shape
# no activation(linear)
    decoder_reye.add(layers.Dense(32*32*1))

    # Models for training and evaluation (prediction)
    model = models.Model(x, [out_caps, decoder_leye(digitcaps), decoder_reye(digitcaps)])
    """
    eval_model = models.Model(x, [out_caps, decoder(digitcaps)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model
    """
    return model

# fit_generator用のbatchを返すgeneratorとバッチ数
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
                yield batch[0], [batch[3], batch[1], batch[2]]
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
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=['mean_absolute_error', 'mean_squared_error', 'mean_squared_error'],
                  loss_weights=[1., args.lam_recon, args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

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
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()



def load_GazeCapture():
    dataset_path = "/home/ohta/workspace/eye_tracking/data"
    train_path = join(dataset_path, 'train')
    x_train = np.load(join(train_path, 'input.npz'))['arr_0']
    y_train = np.load(join(train_path, 'label.npz'))['arr_0']
    eye_train = np.load(join(train_path, 'eyes.npz'))['arr_0']

    test_path = join(dataset_path, 'test')
    x_test = np.load(join(test_path, 'input.npz'))['arr_0']
    y_test = np.load(join(test_path, 'label.npz'))['arr_0']
    eye_test = np.load(join(test_path, 'eyes.npz'))['arr_0']
    print(x_train.shape)
    return (x_train, eye_train, y_train), (x_test, eye_test, y_test)


if __name__ == "__main__":
    import os
    from os.path import join
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--batch_dir', default='/home/docker/share/eye_tracking/data/')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
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
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    #(x_train, y_train), (x_test, y_test) = load_mnist()
    #(x_train, eye_train, y_train), (x_test, eye_test, y_test) = load_GazeCapture()

    # define model
    model = CapsNet(input_shape=(128, 128, 1),
                    n_class=3,
                    routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, args=args)
    """
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)

def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


    """
