import os
import sys
import sys
# import subprocess
# subprocess.run()
import sys

weight_path = "/home/atticus/Projects/Fewshot_Detection/backup/metatunetest1_novel0_neg0/"
result_path = "/home/atticus/Projects/Fewshot_Detection/results/metatunetest1_novel0_neg0/"
weight_names = os.listdir(weight_path)
cur_weight = ""
weight_names.sort(key=lambda x: int(x[:-8]))
for i in range(24, 25):
# for i in range(1):
    cur_weight = weight_path + weight_names[i]
    cmd1 = f"python valid_ensemble.py cfgs/metatune.data cfgs/darknet_dynamic.cfg cfgs/reweighting_net.cfg" \
           f" {cur_weight}"
    os.system(cmd1)
    eval_path = result_path + "ene" + weight_names[i].split(".")[0] + "/comp4_det_test_"
    cmd2 = f"python scripts/voc_eval.py {eval_path}"
    os.system(cmd2)

# sys.stdout.out.close()

