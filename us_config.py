from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 36
config.TRAIN.lr_init = 1e-3
config.TRAIN.lr_gan = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 5
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 20
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = 'data/train_croplabel/'
config.TRAIN.lr_img_path = 'data/train_croptrain/'
## inti_train set location
config.TRAIN.inithr_img_path = 'data/train_croplabel/'
config.TRAIN.initlr_img_path = 'data/train_croptrain/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = './data/test_croplabel/'
config.VALID.lr_img_path = './data/test_croptrain/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
