"""
KTH action recognition under injected camera motion: ConvLSTM vs FEConvLSTM vs MEConvLSTM.

Keller's FERNN KTH protocol (32x32 grayscale, 16 frames at step 2, person split, circular
translations), reusing the Motion-Only MNIST classifier (head, precise BatchNorm, best-val
checkpoint). Camera motion: none, constant, the Moving MNIST velocity processes, or a periodic
shake. See train_kth.py.
"""
