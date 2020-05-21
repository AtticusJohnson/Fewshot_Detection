# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
#python voc/voc_label.py
#cat voc/2007_train.txt voc/2007_val.txt voc/2012_*.txt > voc/voc_train.txt
#python voc/voc_label_1c.py
#python scripts/convert_fewlist.py
python train_meta.py cfgs/metayolo.data cfgs/darknet_dynamic.cfg cfgs/reweighting_net.cfg 000080.weights


