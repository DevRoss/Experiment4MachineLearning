# coding: utf-8
import tensorflow as tf

# 神经网络
INPUT_NODE = 784
OUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# First conv size and depth
CONV1_SIZE = 5
CONV1_DEPTH = 32

# Second conv size and depth
CONV2_SIZE = 5
CONV2_DEPTH = 64

# Full connected neural network
FC_SIZE = 512
