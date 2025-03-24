from typing import Iterator, Tuple, Any

import os
import glob
import random
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

DATASET_PATH = 'dataset/'


class LegoDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'third_person_image': tfds.features.Image(
                            shape=(180, 320, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Third person camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(180, 320, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Robot Cartesian state',
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Gripper position statae',
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Joint position state'
                        )
                    }),
                    'action_dict': tfds.features.FeaturesDict({
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Commanded Cartesian position'
                        ),
                        'cartesian_delta': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Commanded Cartesian delta'
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Commanded gripper position'
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Commanded joint position'
                        ),
                        'joint_delta': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Commanded joint delta'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [6x cartesian deltas, \
                            1x gripper position].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""

        episode_paths = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if f.startswith('episode_')]

        return {
            'train': self._generate_examples(paths=episode_paths)
        }

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            trajectory_path = os.path.join(episode_path, 'trajectory.h5')

            with h5py.File(trajectory_path, 'r') as episode_h5:

                # assemble episode --> here we're assuming demos so we set reward to 1 at the end
                episode = []
                for i in range(len(episode_h5)):
                    step = episode_h5[f'step_{i}']

                    gripper_position = step['observation']['gripper_position'][()]
                    gripper_position_binarized = 1 if gripper_position > 0.5 else 0
                    gripper_position_array = np.array([gripper_position_binarized], dtype=np.float64)

                    episode.append({
                        'observation': {
                            'third_person_image': step['observation']['third_person_image'][()],
                            'wrist_image': step['observation']['wrist_image'][()],
                            'cartesian_position': step['observation']['cartesian_position'][()],
                            'gripper_position': gripper_position_array,
                            'joint_position': step['observation']['joint_position'][()],
                        },
                        'action_dict': {
                            'cartesian_position': step['observation']['cartesian_position'][()],
                            'cartesian_delta': step['action']['cartesian_position_delta'][()],
                            'gripper_position': gripper_position_array,
                            'joint_position': step['observation']['joint_position'][()],
                            'joint_delta': step['action']['joint_position_delta'][()],
                        },
                        'action': np.concatenate(
                            (step['action']['cartesian_position_delta'][()], 
                            gripper_position_array), dtype=np.float64
                        ),
                        'discount': 1.0,
                        'reward': float(i == (len(episode_h5) - 1)),
                        'is_first': i == 0,
                        'is_last': i == (len(episode_h5) - 1),
                        'is_terminal': i == (len(episode_h5) - 1),
                        'language_instruction': step['instruction'][()],
                    })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # for smallish datasets, use single-thread parsing
        for sample in paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

