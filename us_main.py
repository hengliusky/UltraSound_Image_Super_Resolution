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
lr_gan = config.TRAIN.lr_gan
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/Ultrasound_ginit"
    save_dir_gan = "samples/Ultrasound_gan"
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "Us_checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))

    train_inithr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.inithr_img_path, regx='.*.png', printable=False))
    train_initlr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.initlr_img_path, regx='.*.png', printable=False))

    # print(len(train_hr_img_list))
    with tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()
        g_tower_grads = []
        d_tower_grads = []
        g_init_tower_grads = []

        t_image = tf.placeholder(
            'float32', [batch_size * 2, 64, 64, 3],
            name='HR_image')
        t_target_image = tf.placeholder(
            'float32', [batch_size * 2, 64, 64, 3],
            name='LR_image')

        g_init_optim = tf.train.AdamOptimizer(lr_init, beta1=beta1)
        g_optim = tf.train.AdamOptimizer(lr_gan, beta1=beta1)
        d_optim = tf.train.AdamOptimizer(lr_gan, beta1=beta1)

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(2):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        _x = t_image[i * batch_size:(i + 1) *
                                                    batch_size]
                        _y = t_target_image[i * batch_size:(i + 1) *
                                                           batch_size]

                        # ========================== encoder ============================
                        net_g = UsGan_g(_x, is_train=True, reuse=False)
                        t_recon = net_g.outputs
                        # print('The param of generator is %d' % net_g.count_params())
                        net_d, dis_logits_true = UsGan_d(_y, reuse=False)
                        _, dis_logits_fake = UsGan_d(t_recon, reuse=True)

                        ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
                        t_target_image_224 = tf.image.resize_images(_y, size=[224, 224], method=0,
                                                                    align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
                        t_predict_image_224 = tf.image.resize_images(t_recon, size=[224, 224], method=0,
                                                                     align_corners=False)  # resize_generate_image_for_vgg

                        net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
                        _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)
                        # print('The param of discriminator is %d' % net_d.count_params())
                        tf.get_variable_scope().reuse_variables()

                        vgg_loss = tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs,
                                                                        is_mean=True)
                        l2_loss = tl.cost.mean_squared_error(t_recon, _y, is_mean=True)
                        # l1_loss = tl.cost.absolute_difference_error(t_recon, _y, is_mean=True)

                        g_gan_loss = tl.cost.sigmoid_cross_entropy(
                            dis_logits_fake,
                            tf.ones_like(dis_logits_fake),
                            name='g')

                        d_loss_real = tl.cost.sigmoid_cross_entropy(
                            dis_logits_true,
                            tf.ones_like(dis_logits_true),
                            name='d1')
                        d_loss_fake = tl.cost.sigmoid_cross_entropy(
                            dis_logits_fake,
                            tf.zeros_like(dis_logits_fake),
                            name='d2')
                        # ==================================================================

                        g_loss = 1e-4 * g_gan_loss + 2e-6 * vgg_loss + l2_loss
                        g_init_loss = l2_loss
                        d_loss = d_loss_real + d_loss_fake

                        param = tf.trainable_variables()
                        G_params = [i for i in param if 'UsGan_g' in i.name]
                        D_params = [
                            i for i in param if 'UsGan_d' in i.name
                        ]

                        g_grads = g_optim.compute_gradients(
                            g_loss, var_list=G_params)
                        g_tower_grads.append(g_grads)

                        g_init_grads = g_init_optim.compute_gradients(
                            g_init_loss, var_list=G_params)
                        g_init_tower_grads.append(g_init_grads)

                        d_grads = d_optim.compute_gradients(
                            d_loss, var_list=D_params)
                        d_tower_grads.append(d_grads)

                        with tf.name_scope('Image'):
                            tf.summary.image(
                                'reconstruction', t_recon, max_outputs=3)
                            tf.summary.image('input', _x, max_outputs=3)
                            tf.summary.image('target', _y, max_outputs=3)
                        with tf.name_scope('g_Loss'):
                            tf.summary.scalar('g_loss', g_loss)
                            tf.summary.scalar('g_gan_loss', g_gan_loss)
                            tf.summary.scalar('l2_loss', l2_loss)
                            # tf.summary.scalar('feature_loss', feature_loss)
                        with tf.name_scope('d_Loss'):
                            tf.summary.scalar('d_loss', d_loss)
                            tf.summary.scalar('d_loss_real', d_loss_real)
                            tf.summary.scalar('d_loss_fake', d_loss_fake)

        grads_g = average_gradients(g_tower_grads)
        train_g = g_optim.apply_gradients(grads_g, global_step=global_step)
        grads_init_g = average_gradients(g_init_tower_grads)
        train_init_g = g_optim.apply_gradients(grads_init_g, global_step=global_step)
        grads_d = average_gradients(d_tower_grads)
        train_d = d_optim.apply_gradients(grads_d, global_step=global_step)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)

        # # ========================== RESTORE MODEL =============================
        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False))
        loss_writer = tf.summary.FileWriter('./train', sess.graph)
        sess.run(tf.global_variables_initializer())

        ## SRGAN
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)
        # grads, variables = zip(*optimizer.compute_gradients(g_loss))
        # grads, global_norm = tf.clip_by_global_norm(grads, 10)
        # g_optim = optimizer.apply_gradients(zip(grads, variables))

        ###============================= LOAD VGG ===============================###
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()

        params = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        tl.files.assign_params(sess, params, net_vgg)
        # net_vgg.print_params(False)
        # net_vgg.print_layers()
        merged = tf.summary.merge_all()
        count = len(train_hr_img_list) // (batch_size * 2)
        ###============================= TRAINING ===============================###
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_Us_init10.npz',
        #                                  network=net_g)
        ###========================= initialize G ====================###
        # fixed learning rate
        print(" ** fixed learning rate: %f (for init G)" % lr_init)
        for epoch in range(0, n_epoch_init+1):
            epoch_time = time.time()
            total_mse_loss, n_iter = 0, 0

            ## If your machine cannot load all images into memory, you should use
            ## this one to load batch of images while training.
            # random.shuffle(train_hr_img_list)
            # for idx in range(0, len(train_hr_img_list), batch_size):
            #     step_time = time.time()
            #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
            #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
            #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

            ## If your machine have enough memory, please pre-load the whole train set.
            for idx in range(0, count):
                step_time = time.time()
                b_imgs_384 = tl.prepro.threading_data(
                    train_inithr_img_list[idx : idx + batch_size * 2],
                        fn=get_imgs_fn, path=config.TRAIN.inithr_img_path)
                b_imgs_96 = tl.prepro.threading_data(
                    train_initlr_img_list[idx : idx + batch_size * 2],
                        fn=get_imgs_fn, path=config.TRAIN.initlr_img_path)

                ## update G
                errM, _ = sess.run([g_init_loss, train_init_g], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
                n_iter += 1
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
            print(log)

            ## save model
            if (epoch != 0) and (epoch % 5 == 0):
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_USinit_%s.npz' % epoch, sess=sess)


        ###========================= train GAN (SRGAN) =========================###
        # count = len(train_hr_img_list)//batch_size
        for epoch in range(0, n_epoch+1):
            ## update learning rate
            if epoch !=0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(lr_v,  * new_lr_decay))
                log = " ** new learning rate: %f (for GAN)" % (lr_gan * new_lr_decay)
                print(log)
            elif epoch == 0:
                sess.run(tf.assign(lr_v, lr_gan))
                log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_gan, decay_every, lr_decay)
                print(log)

            epoch_time = time.time()
            total_d_loss, total_g_loss, n_iter = 0, 0, 0

            ## If your machine cannot load all images into memory, you should use
            ## this one to load batch of images while training.
            # random.shuffle(train_hr_img_list)
            # for idx in range(0, len(train_hr_img_list), batch_size):
            #     step_time = time.time()
            #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
            #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
            #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

            ## If your machine have enough memory, please pre-load the whole train set.
            for idx in range(0, count):
                step_time = time.time()
                b_imgs_384 = tl.prepro.threading_data(
                    train_hr_img_list[idx : idx + batch_size * 2],
                        fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
                b_imgs_96 = tl.prepro.threading_data(
                    train_lr_img_list[idx : idx + batch_size * 2],
                        fn=get_imgs_fn, path=config.TRAIN.lr_img_path)
                ## update D
                if (count * epoch + idx)%5 == 0:
                    errD, summary, _ = sess.run([d_loss, merged, train_d], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                    loss_writer.add_summary(summary, count * epoch + idx)
                ## update G
                errG, errM, errV, errA, _ = sess.run([g_loss, l2_loss, vgg_loss, g_gan_loss, train_g], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                loss_writer.add_summary(summary, count * epoch + idx)
                print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
                total_d_loss += errD
                total_g_loss += errG
                n_iter += 1

            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
            print(log)

            ## save model
            if (epoch != 0) and (epoch % 1 == 0):
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_US%s.npz' % epoch, sess=sess)
                tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_US%s.npz' % epoch, sess=sess)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    train()
