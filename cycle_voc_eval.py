import os
from tqdm import tqdm 

root_path = os.getcwd()
filePath = root_path + '/backup/metayolo_novel0_neg1/'
restore_files = os.listdir(filePath)
if not os.path.exists("/input/program/Fewshot_Detection/results/"):
    os.mkdir("/input/program/Fewshot_Detection/results/")
for i in tqdm(range(len(restore_files))):
    restore_file = restore_files[i]
    os.system("python valid_ensemble.py cfgs/metayolo.data cfgs/darknet_dynamic.cfg cfgs/reweighting_net.cfg {}"
              .format(filePath+restore_file))
#     os.system("python scripts/voc_eval.py {}".format("/input/results/{}_".format(restore_file[:-8])))
    os.system("python scripts/voc_eval.py /input/program/Fewshot_Detection/results/metayolo_novel0_neg1/ene{0}/comp4_det_test_".format(restore_file[:-8]))
