1.For opts.py, if run on kitti/bdd, copy the content of opts_kitti_bdd.py to opts.py,
and if run on MOSE, copy the content of opts_for_MOSE.py to opts.py,
and copy the content of deep_sort/tracker_for_MOSE.py to deep_sort/tracker.py,
because DeepSORT uses the opts.py and deep_sort tracker.py.
2.If run on ytvis 2019 dataset, copy the content of opts_for_ytvis_2019.py to opts.py,
and use strong_sort_multi_class_ytvis.py, instead strong_sort.py to deal with many categories.
The command is like:
python strong_sort_multi_class_ytvis.py YouTube_VIS_2019 valid
