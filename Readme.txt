This is a remind for config.
[0] Complete the data and checkpoint(VOCdevkit, checkpoints, backup).
[1] Generate label:
    python voc_label.py 
    python voc_label_1c.py
[2] Train basic network:
    python train_meta.py cfgs/metayolo.data cfgs/darknet_dynamic.cfg cfgs/reweighting_net.cfg darknet19_448.conv.23
[3] Eval basic network:
    python valid_ensemble.py cfgs/metayolo.data cfgs/darknet_dynamic.cfg cfgs/reweighting_net.cfg backup/metayolo_novel0_neg1/000170.weights
    python scripts/voc_eval.py results/metayolo_novel0_neg1/ene000170/comp4_det_test_
