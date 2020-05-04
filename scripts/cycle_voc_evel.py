import os

root_path = os.getcwd()
filePath = root_path + '/backup/metayolo_novel0_neg1/'
restore_files = os.listdir(filePath)
if not os.path.exists("/input/results/"):
    os.mkdir("/input/results/")
for restore_file in restore_files:
    os.system("python valid_ensemble.py cfgs/metayolo.data cfgs/darknet_dynamic.cfg cfgs/reweighting_net.cfg {}"
              .format(filePath+restore_file))
    os.system("python scripts/voc_eval.py results/comp4_det_test_")