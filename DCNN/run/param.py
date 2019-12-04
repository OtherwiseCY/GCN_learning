import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

param = {
    "n_hop":3,
    "feat_shape":1433,
    "num_node":2708,
    "num_class":7,
    "loss_fn":nn.CrossEntropyLoss(),
    "optim":optim.SGD,
    "epoch":200,
    "lr":.05,
    "earlystopping":.001,
    "earlystopround":20,
    "train_val_test_ratio":(8,1,1),
    "momentum":.9
}