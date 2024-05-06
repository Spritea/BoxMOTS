1.hooks.py is copied from /home/wensheng/anaconda3/envs/adelaidet_mose/lib/python3.8/site-packages/detectron2/engine/train_loop.py,
without any modification.
2.train_loop.py is copied from /home/wensheng/anaconda3/envs/adelaidet_mose/lib/python3.8/site-packages/detectron2/engine/hooks.py,
without any modification.
3.my_defaults.py is modified from /home/wensheng/anaconda3/envs/adelaidet_mose/lib/python3.8/site-packages/detectron2/engine/defaults.py.
to change the class DefaultPredictor to output reid feat.
4.hooks.py and train_loop.py and the empty __init__.py is copied for the import of my_defaults.py.