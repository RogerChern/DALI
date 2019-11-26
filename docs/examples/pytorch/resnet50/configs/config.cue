data_dir               : "/mnt/lustre/chenyuntao1/datasets/imagenet"
_architecture          : "resnet50"
arch                   : _architecture
workers                : 4

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
}

val_batch_size         : 25
print_freq             : 10
_train                 : mode == "train"
_evaluate              : mode == "val"
_feature_extract       : mode == "extract"
fp16                   : false
dali_cpu               : false
if !dynamic_loss_scale {
static_loss_scale      : 128
}
dynamic_loss_scale     : false
_profile               : false
prof                   : _profile
_test                  : mode == "test"

if _train {
train_list             : "/mnt/lustre/chenyuntao1/datasets/imagenet/train.lst"
train_map              : "/mnt/lustre/chenyuntao1/datasets/imagenet/train.map"
validation_start_epoch : -50
}

if _train || _evaluate {
val_list               : "/mnt/lustre/chenyuntao1/datasets/imagenet/val.lst"
val_map                : "/mnt/lustre/chenyuntao1/datasets/imagenet/val.map"
}

if _train {
nag                    : false
cos_lr                 : false
}

mode                   : "train" | "val" | "extract" | "test"
mode                   : "val"
