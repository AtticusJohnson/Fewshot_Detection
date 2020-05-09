import torch
import sys
sys.path.append("..")
from darknet_meta import Darknet
from region_loss import RegionLossV2
import tensorwatch as tw
# from tensorboardX import SummaryWriter

# writer = SummaryWriter('./myCode/Result')
darknetcfg    = "D:\Code\python\Fewshot_Detection\cfgs\darknet_dynamic.cfg"
learnetcfg    = "D:\Code\python\Fewshot_Detection\cfgs/reweighting_net.cfg"

model = Darknet(darknetcfg, learnetcfg)
region_loss = model.loss

# 查看网络结构
model.print_network()
data_shape = (1, 3, 448, 448)
mask_shape = (5, 1, 448, 448)
meta_shape = (5, 3, 448, 448)
target_shape = (1, 5, 250)
# model.train()

#model测试
origin_data = torch.randn(data_shape)
mask = torch.where(torch.randn(mask_shape) > 0, torch.ones(mask_shape), torch.zeros(mask_shape))
meta_data = torch.where(torch.randn(meta_shape) > 0, torch.ones(meta_shape), torch.zeros(meta_shape))
out = model(origin_data, meta_data, mask)
# tw.draw_model(model, (data_shape, meta_shape, mask_shape), "./model.png")
# writer.add_graph(model, (origin_data, meta_data, mask))
# writer.close()
# print(out.shape)
# head测试
# target = torch.ones(target_shape) * 0.1
# loss = region_loss(out, target)



