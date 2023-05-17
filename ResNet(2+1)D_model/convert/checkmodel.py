import torch
from torchsummary import summary
# model = torch.load("F:/中国手语数据集科大/SLR/cnn3d_models/10分类 参数/slr_cnn3d_epoch010.pth")
model = torch.load("F:/中国手语数据集科大/SLR/cnn3d_models/10分类 非参数/slr_cnn3d_epoch005.pth")
summary(model, (1, 3, 6, 128, 128))