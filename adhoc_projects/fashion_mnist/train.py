'''Training the model & testing on Test set in this Script'''

import numpy as np
import time, sys, os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

import dataload, model

def print_function(i, num_batches, acc_temp):
    j = int(i/25)
    sys.stdout.write('\r{0}'.format('Batches: ' + str(i+1) + '/' + str(num_batches) +\
                                    ' [' + '='*j + '>' + (int((num_batches)/25)-j) * ' ' + '] ' + \
                                    str(round((i+1)*100/num_batches)) + '%; Accuracy: ' + \
                                    str(acc_temp)))
    sys.stdout.flush()
    pass

def data_augmentation(loader):
    t_loader = []
    
    for i, (batch_x, batch_y) in enumerate(loader):
        sess = tf.Session()
        num_sample = len(batch_x)
        aug_size = int(0.05*num_sample)
        idx = np.random.choice(np.arange(num_sample), replace=False, size=aug_size)
        b_x = batch_x[idx]
        b_y = batch_y[idx]

        rot_img = tf.image.flip_left_right(b_x.reshape((aug_size, 28, 28)))
        b_x = sess.run(rot_img).reshape((aug_size, 28 * 28))
        batch_x = np.vstack((batch_x, b_x))
        batch_y = np.vstack((batch_y, b_y))

        idx = np.random.choice(np.arange(num_sample), replace=False, size=aug_size)
        b_x = batch_x[idx]
        b_y = batch_y[idx]
        rot_img = tf.image.random_flip_up_down(b_x.reshape((aug_size, 28, 28)))
        b_x = sess.run(rot_img).reshape((aug_size, 28 * 28))
        batch_x = np.vstack((batch_x, b_x))
        batch_y = np.vstack((batch_y, b_y))
        
        idx = np.random.choice(np.arange(num_sample), replace=False, size=aug_size)
        b_x = batch_x[idx]
        b_y = batch_y[idx]
        rot_img = tf.image.random_flip_up_down(b_x.reshape((aug_size, 28, 28)))
        rot_img = tf.image.random_flip_left_right(rot_img, seed = None)
        b_x = sess.run(rot_img).reshape((aug_size, 28 * 28))
        batch_x = np.vstack((batch_x, b_x))
        batch_y = np.vstack((batch_y, b_y))

        t_loader.append((batch_x, batch_y))
    sess.close()
    return t_loader

def train(args = None):
    epochs = args.epochs
    validation = True
    train_stats, valid_stats = [], []
    train_stats_l, valid_stats_l = [], []
    
    if opt == 1:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(model.loss)
    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(model.loss)
    
    saver = tf.train.Saver(tf.trainable_variables())

    config = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6,
                            device_count = {'CPU': 6})
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Training Started!!')
        
        num_batches = len(train_loader)
        best_valid_acc = 0
        best_valid_loss = 1e+10
        early_limit = 0

        for e in range(epochs):
            if early_limit > args.patience:
                print('Early stop! No improvement in Validation Loss since %s Epochs!'%(args.patience))
                print('-'*65)
                break
            
            print('Epoch: ', e+1)
            start = time.time()
            valid_score = 0
            train_accuracies, train_loss = [], []

            for i, (batch_x, batch_y) in enumerate(train_loader):
                num_sample = len(batch_x)
                
                acc_temp = round(np.mean(train_accuracies), 4) if len(train_accuracies) != 0 else 'NA'
                print_function(i, num_batches, acc_temp)

                _, acc, loss = sess.run([optimizer, model.accuracy, model.loss],
                                        feed_dict={x: batch_x, y: batch_y})
                train_accuracies.append(acc)
                train_loss.append(loss)
            acc = np.mean(train_accuracies)
            loss_train = np.mean(train_loss)
            t = time.time() - start
            print('\nTime: %s sec; Traning Accuracy %.3f; Training Loss: %.3f'%(round(t), acc, loss_train))

            #Validation
            if validation:
                valid_accuracies, valid_loss = [], []
                for i, (batch_x, batch_y) in enumerate(valid_loader):
                    acc, loss = sess.run([model.accuracy, model.loss], feed_dict = {x: batch_x, y: batch_y})
                    valid_accuracies.append(acc)
                    valid_loss.append(loss)
                valid_acc, loss_valid = np.mean(valid_accuracies), np.mean(valid_loss)
                print('Validation Accuracy %.3f; Validation Loss: %.3f'%(valid_acc, loss_valid))
                if valid_acc > best_valid_acc:
                    acc = int(round(valid_acc*100)); best_valid_acc = valid_acc
                    print('Best Validation Accuracy achieved at Epoch: %s is %.3f'%(e+1, best_valid_acc))
                    ckpt_file = os.path.join(model_dir, 'best_model_da.ckpt')
                    saver.save(sess, ckpt_file)
                if loss_valid < best_valid_loss:
                    best_valid_loss = loss_valid
                    early_limit = 0
            print('-'*65)
            early_limit = early_limit + 1
            train_stats.append(train_accuracies); train_stats_l.append(train_loss)
            valid_stats.append(valid_accuracies); valid_stats_l.append(valid_loss)
        print('Training Completed!!'); print('-'*65)
    return train_stats, train_stats_l, valid_stats, valid_stats_l
                                             
def test(args = None):
    config = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6,
                            device_count = {'CPU': 6})
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session(config = config) as sess:
        sess.run(tf.local_variables_initializer())
        print('Testing the model on 10000 Images!')
        ckpt_file = os.path.join(model_dir, best_val_model)
        saver.restore(sess, ckpt_file)

        test_predicted = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            correct = sess.run([model.correct_predicted], feed_dict = {x: batch_x, y: batch_y})
            test_predicted.append(np.sum(correct))
        test_acc = np.sum(test_predicted)/num_samples_test
        print('Testing Accuracy %.3f'%(test_acc))
    pass

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate for Optimiser')
parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size for mini-batch Opitimisation')
parser.add_argument('-s', '--save_dir', type=str, default='./models/', help='directory to save the model')
parser.add_argument('-o', '--opt', type=int, default=1, help='1 for Adam, 2 for RMSProp')
parser.add_argument('-d', '--data_aug', type=int, default=2, help='1 for True, 2 for False')
parser.add_argument('-t', '--test', type=int, default=0, help='1 for test, 0 for train')
parser.add_argument('-e', '--epochs', type=int, default=40, help='Number of Epochs')
parser.add_argument('-p', '--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('-k', '--filter_size', type=int, default = 5, help='filter_size to be used for the model')

args = parser.parse_args()

batch_size = args.batch_size
model_dir = args.save_dir
testing = args.test
lr = args.lr
opt = args.opt
mode = 'train'

Loader = dataload.DataLoader()
x, y = Loader.load_data()

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.1)

num_classes = len(np.unique(y_train))

x = tf.placeholder(dtype=tf.float32, shape = (None, x_train.shape[1]))
y = tf.placeholder(dtype=tf.float32, shape = (None, num_classes))

if args.filter_size == 5:
    model = model.CNN(x, y)
else:
    model = model.CNN3(x, y)

best_val_model = 'best_model_da.ckpt'

if testing:
    x_test, y_test = Loader.load_data(mode = 'test')
    test_loader = Loader.create_batches(x_test, y_test, batch_size = batch_size)
    num_samples_test = x_test.shape[0]
    
    test(args)
else:
    if args.data_aug == 1:
        print('Data Augmentation started!!')
        start = time.time()
        loader = Loader.create_batches(x_train, y_train, batch_size = batch_size)
        train_loader = data_augmentation(loader)
        print('Data Augmentation complete! Time Elapsed: %s sec'%(int(time.time()-start)))
    else:
        train_loader = Loader.create_batches(x_train, y_train, batch_size = batch_size)
        
    valid_loader = Loader.create_batches(x_valid, y_valid, batch_size = batch_size)

    train_stats, train_stats_l, valid_stats, valid_stats_l = train(args)
    epoch_loss_t = np.mean(train_stats_l, axis = 1)
    epoch_loss_v = np.mean(valid_stats_l, axis = 1)

    num_epochs = np.arange(1, len(epoch_loss_t)+1)

    plt.plot(num_epochs, epoch_loss_t, marker = 'x')
    plt.plot(num_epochs, epoch_loss_v, marker = 'o')
    plt.xlabel('Epoch number')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss vs Epoch')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()