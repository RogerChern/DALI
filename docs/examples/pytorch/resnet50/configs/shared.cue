package config

data_dir               : "/mnt/lustre/chenyuntao1/datasets/imagenet"
_architecture          : "resnet50"
arch                   : _architecture
workers                : 4
_train                 : mode == "train"
_evaluate              : mode == "val"
_feature_extract       : mode == "extract"
_test                  : mode == "test"
_profile               : false
prof                   : _profile
fp16                   : false
dali_cpu               : false
print_freq             : 10

mode                   : "train" | "val" | "extract" | "test"
mode                   : "train"