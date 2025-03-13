import argparse
import tqdm
import importlib
import os
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', help='name of the dataset to visualize')
args = parser.parse_args()

# create TF dataset
dataset_name = args.dataset_name
print(f"Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, split='train')
#ds = ds.shuffle(100)

# visualize episodes
for i, episode in enumerate(ds):
    third_person_image, wrist_image, = [], []
    for step in episode['steps']:
        third_person_image.append(step['observation']['third_person_image'].numpy())
        wrist_image.append(step['observation']['wrist_image'].numpy())

    third_person_strip = np.concatenate(third_person_image[::4], axis=1)
    wrist_strip = np.concatenate(wrist_image[::4], axis=1)
    image_strip = np.concatenate((third_person_strip, wrist_strip), axis=0)
    caption = step['language_instruction'].numpy().decode() + ' (temp. downsampled 4x)'

    plt.figure()
    plt.imshow(image_strip)
    plt.title(caption)

# visualize action and state statistics
actions, states = [], []
for episode in tqdm.tqdm(ds.take(500)):
    for step in episode['steps']:
        actions.append(step['action'].numpy())
        states.append(step['observation']['cartesian_position'].numpy())
actions = np.array(actions)
states = np.array(states)
action_mean = actions.mean(0)
state_mean = states.mean(0)

def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(5*n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem+1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(vector_mean[elem])

vis_stats(actions, action_mean, 'action_stats')
vis_stats(states, state_mean, 'state_stats')


plt.show()