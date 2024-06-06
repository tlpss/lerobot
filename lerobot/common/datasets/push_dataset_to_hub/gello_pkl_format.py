#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process pickle files formatted like in: https://github.com/fyhMer/fowm"""

import pickle
import shutil
from pathlib import Path

import einops
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames




def load_from_raw(raw_dir, out_dir, fps, video, debug):

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    episode_paths = list(raw_dir.glob("*"))
    episode_paths = [x for x in episode_paths if x.is_dir()]
    episode_paths = sorted(episode_paths)


    dataset_idx = 0
    episode_dicts = []
    episode_data_index = {}
    episode_data_index["from"] = []
    episode_data_index["to"] = []
    for episode_idx, episode_path in enumerate(episode_paths):
        step_paths = list(episode_path.glob("*.pkl"))
        step_paths = sorted(step_paths)
    
        episode_dict = {}
        for key in pickle.load(open(step_paths[0], "rb")).keys():
            episode_dict[key] = []

        for key in ["frame_index", "episode_index", "index", "timestamp", "next.done", "next.success"]:
            episode_dict[key] = []
        
        episode_start_idx = dataset_idx
        for step_idx, step_path in enumerate(step_paths):
            step_dict = pickle.load(open(step_path, "rb"))

            for key in pickle.load(open(step_path, "rb")).keys():
                episode_dict[key].append(step_dict[key])
            
            episode_dict["frame_index"].append(step_idx)
            episode_dict["episode_index"].append(episode_idx)
            episode_dict["index"].append(step_idx)
            episode_dict["timestamp"].append(step_idx / fps)

            # assume all demonstrations are successful at the last step.
            episode_dict["next.done"].append(len(step_paths) - 1 == step_idx)
            episode_dict["next.success"] = episode_dict["next.done"]

            dataset_idx += 1
        
        episode_data_index["from"].append(episode_start_idx)
        episode_data_index["to"].append(dataset_idx-1)

        # rename the keys to match the expected format       

        episode_dict["action"] = episode_dict["control"]
        episode_dict.pop("control")


        for key in list(episode_dict.keys()):
            if "rgb" in key or "depth" in key:
                # parse the key
                cam_name, img_type = key.split("_")
                episode_dict[f"observation.images.{cam_name}.{img_type}"] = episode_dict[key]

                # drop the original key
                episode_dict.pop(key)

        # convert to the desired formats
        for key in episode_dict.keys():
            if "rgb" in key or "depth" in key:
                # convert to uint8
                episode_dict[key] = [x.astype("uint8") for x in episode_dict[key]]
                if "depth" in key:
                    episode_dict[key] = [x[..., 0] for x in episode_dict[key]]

                if video:
                    # save png images in temporary directory
                    tmp_imgs_dir = out_dir / "tmp_images"
                    save_images_concurrently(episode_dict[key], tmp_imgs_dir)
                    # encode images to a mp4 video
                    fname = f"{key}_episode_{episode_idx:06d}.mp4"
                    video_path = out_dir / "videos" / fname
                    encode_video_frames(tmp_imgs_dir, video_path, fps)

                    # clean temporary images directory
                    shutil.rmtree(tmp_imgs_dir)

                    # store the reference to the video frame
                    episode_dict[key] = [{"path": f"videos/{fname}", "timestamp": i / fps} for i in range(len(episode_dict[key]))]

                else:
                    episode_dict[key] = [PILImage.fromarray(x) for x in episode_dict[key]]


            else:
                episode_dict[key] = torch.tensor(episode_dict[key])



        
        episode_dicts.append(episode_dict)

        # state = joint_positions + gripper_position
        episode_dict["observation.state"] = episode_dict["joint_positions"] + episode_dict["gripper_position"]
        episode_dict.pop("joint_positions")
        episode_dict.pop("gripper_position")

        # for now, drop all depth images

        for key in list(episode_dict.keys()):
            if "depth" in key:
                episode_dict.pop(key)
        

        # for now drop tcp_pose_quat and wrench
        episode_dict["tcp_pose_rotvec"] = episode_dict["tcp_pose_quat"]
        episode_dict.pop("tcp_pose_quat")

        episode_dict.pop("wrench")

        if debug:
            break
        
    data_dict = concatenate_episodes(episode_dicts)
    return data_dict, episode_data_index



def to_hf_dataset(data_dict, video):
    features = {}

    keys = [key for key in data_dict.keys() if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)
    features["next.success"] = Value(dtype='bool', id=None)

    features["tcp_pose_rotvec"] = Sequence(Value(dtype="float32", id=None), length=6)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=True, debug=False):
    if fps is None:
        fps = 10

    data_dict, episode_data_index = load_from_raw(raw_dir, out_dir, fps, video, debug)
    hf_dataset = to_hf_dataset(data_dict, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
