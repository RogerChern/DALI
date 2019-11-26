package config

if _train {
epochs                 : 90
warmup_epochs          : 1 + epochs quo 20
lr_step_def            :: < epochs & > 0
lr_steps               : [...lr_step_def]
lr_steps               : [30, 60, 80]
start_epoch_def        :: < epochs
start_epoch            : start_epoch_def & 0
batch_size             : 256
lr                     : 0.1
momentum               : 0.9
weight_decay           : 1e-4
no_bn_wd               : false
zero_init_resblock     : false
resume                 : false
pretrain               : false
if !dynamic_loss_scale {
static_loss_scale      : 128
}
dynamic_loss_scale     : false
val_batch_size         : 25
train_list             : "/mnt/lustre/chenyuntao1/datasets/imagenet/train.lst"
train_map              : "/mnt/lustre/chenyuntao1/datasets/imagenet/train.map"
val_list               : "/mnt/lustre/chenyuntao1/datasets/imagenet/val.lst"
val_map                : "/mnt/lustre/chenyuntao1/datasets/imagenet/val.map"
validation_start_epoch : -50
nag                    : false
cos_lr                 : false
}