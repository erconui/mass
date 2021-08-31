#!/usr/bin/python3
import numpy as np

target_list = np.load("target_list.npy")
target_list_time = np.load("target_list_time.npy")

target_loc = target_list[0][0][:2]
print(target_loc)

tracks_list = np.load("tracks_list.npy")
tracks_list_time = np.load("tracks_list_time.npy")
