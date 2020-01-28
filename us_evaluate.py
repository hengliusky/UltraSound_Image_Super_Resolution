#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from us_model1 import *
from us_utils import *
from us_config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx:idx + n_threads]
        b_imgs = tl.prepro.threading_data(
            b_imgs_list, fn=get_imgs_fn, path=path)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def evaluate():
    ## create folders to save result images
    save_dir = "samples/our2"
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "/media/ahut207/191a195c-cd3b-4250-aa37-7f9cc4bf027b/Ultrasound_SR/Us_checkpoint/zxy/l1andssim2/"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.*', printable=False))

    valid_hr_imgs = tl.prepro.threading_data(
            valid_hr_img_list,
            fn=get_imgs_fn, path=config.VALID.hr_img_path)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    t_image = tf.placeholder('float32', [None, None, None, 3], name='input_image')

    net_g = UsGan_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_US50.npz', network=net_g)
    tim = 0
    for imid in range(len(valid_hr_img_list)):
        valid_hr_img = valid_hr_imgs[imid]

        size = valid_hr_img.shape
        valid_lr_img = scipy.misc.imresize(valid_hr_img, [size[0] // 4, size[1] // 4], interp='bicubic', mode=None)
        valid_lr_img = scipy.misc.imresize(valid_lr_img, [size[0], size[1]], interp='bicubic', mode=None)

        print(valid_lr_img.shape)

        valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        out = np.clip(out, -1., 1.)
        tim += time.time() - start_time

        print("took: %4.4fs" % (time.time() - start_time))

        print("LR size: %s /  generated HR size: %s" % (
        valid_lr_img.shape, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir + '/' + valid_hr_img_list[imid])

    print(tim / len(valid_hr_img_list))

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.jpg', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.jpg', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()
    # valid_lr_imgs = tl.prepro.threading_data(
    #     valid_lr_img_list,
    #     fn=get_imgs_fn, path=config.VALID.lr_img_path)
    # valid_hr_imgs = tl.prepro.threading_data(
    #     valid_hr_img_list,
    #     fn=get_imgs_fn, path=config.VALID.hr_img_path)
    # ###========================== DEFINE MODEL ============================###
    # imid = 1  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    # valid_lr_img = valid_lr_imgs[imid]
    # valid_hr_img = valid_hr_imgs[imid]
    # # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    # # valid_lr_img = (valid_lr_img / 255.) # rescale to ［－1, 1]
    # # print(valid_lr_img.min(), valid_lr_img.max())
    #
    # size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
    # # t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    #
    # net_g = UsGan_g(t_image, is_train=False, reuse=False)
    #
    # ###========================== RESTORE G =============================###
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # tl.layers.initialize_global_variables(sess)
    # # tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_US3.npz', network=net_g)
    # tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + 'g_US90.npz', network=net_g)
    # ###======================= EVALUATION =============================###
    # start_time = time.time()
    # out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    # out = np.clip(out, -1., 1.)
    # print("took: %4.4fs" % (time.time() - start_time))
    #
    # print("LR size: %s /  generated HR size: %s" % (
    # size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    # print("[*] save images")
    # tl.vis.save_image(out[0], save_dir + '/valid_gen.jpg')
    # tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.jpg')
    # tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.jpg')

    # out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    # tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    evaluate()
