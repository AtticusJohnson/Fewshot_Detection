import torch
import sys
sys.path.append("..")
from darknet_meta import Darknet
from region_loss import RegionLossV2

darknetcfg    = "D:\Code\python\Fewshot_Detection\cfgs\darknet_dynamic.cfg"
learnetcfg    = "D:\Code\python\Fewshot_Detection\cfgs/reweighting_net.cfg"

model = Darknet(darknetcfg, learnetcfg)
region_loss = model.loss

# 查看网络结构
model.print_network()
data_shape = [8, 3, 448, 448]
mask_shape = [8, 1, 448, 448]
meta_shape = [8, 3, 448, 448]
target_shape = [8, 15, 250]
model.train()

# model测试
origin_data = torch.randn(data_shape)
mask = torch.where(torch.randn(mask_shape) > 0, torch.ones(mask_shape), torch.zeros(mask_shape))
meta_data = torch.where(torch.randn(meta_shape) > 0, torch.ones(meta_shape), torch.zeros(meta_shape))
out = model(origin_data, meta_data, mask)
print(out.shape)

# head测试
target = torch.ones(target_shape) * 0.1
loss = region_loss(out, target)


