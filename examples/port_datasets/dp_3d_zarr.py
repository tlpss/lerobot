import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset

TASK_DESCRIPTION = "Push the T-shaped blue block onto the T-shaped green target surface."
DEFAULT_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (10,),
        "names": {
            "axes": ["x", "y", "z", "rxx", "rxy", "rxz", "ryx", "ryy", "ryz", "gripper_width"],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (10,),
        "names": {
            "axes": ["x", "y", "z", "rxx", "rxy", "rxz", "ryx", "ryy", "ryz", "gripper_width"],
        },
    },
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
}


def load_frames_from_mp4_video(video_path: Path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    # get the fps of the video recording
    frames = []
    timestamps = []
    while cap.isOpened():
        ret, frame = cap.read()
        # get the timestamp of the frame
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ret:
            break
        # bgr to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        timestamps.append(timestamp)
    cap.release()
    return frames, timestamps


def get_synced_frames_from_mp4_video(video_path, timestamps):
    # for each timestamp, get the frame that is closest to the timestamp, but not after
    frames, video_timestamps = load_frames_from_mp4_video(video_path)
    # the video timestamps are in milliseconds since the beginning of the video
    # the timestamps are in seconds, absolute
    relative_timestamps = [timestamp - timestamps[0] for timestamp in timestamps]
    # cvoert to milliseconds
    relative_timestamps = [timestamp * 1000 for timestamp in relative_timestamps]
    synced_frames = []
    for timestamp in relative_timestamps:
        # find the frame that is closest to the timestamp, but stricly before
        frame_idx = np.argwhere(np.array(video_timestamps) <= timestamp).max()
        synced_frames.append(frames[frame_idx])
    return synced_frames


def build_features(mode, n_cameras, img_shape) -> dict:
    features = DEFAULT_FEATURES
    for i in range(n_cameras):
        features[f"observation.image.camera_{i}"] = {
            "dtype": mode,
            "shape": img_shape,
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
    return features


def load_raw_dataset(zarr_path: Path):
    from lerobot.common.datasets.push_dataset_to_hub._diffusion_policy_replay_buffer import (
        ReplayBuffer as DiffusionPolicyReplayBuffer,
    )

    zarr_data = DiffusionPolicyReplayBuffer.copy_from_path(zarr_path)
    return zarr_data


def main(
    raw_dir: Path, repo_id: str, mode: str = "video", push_to_hub: bool = True, n_episodes: int | None = None
):
    if mode not in ["video", "image"]:
        raise ValueError(mode)

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        raise FileNotFoundError(f" Raw data dir {raw_dir} does not exist")

    # find zarr file in the raw dir
    files = list(raw_dir.glob("*.zarr"))
    assert len(files) == 1, f"Expected to find one zarr file in {raw_dir}, but found {files}"

    zarr_data = load_raw_dataset(zarr_path=raw_dir / files[0])

    video_dir = raw_dir / "videos"
    n_camera_views = len(list((video_dir / "0").glob("*.mp4")))
    print(f"Found {n_camera_views} camera views")

    action = zarr_data["action"][:]
    robot_state = zarr_data["robot_eef_pose_6d_rot"][:]
    gripper_width = zarr_data["gripper_width"][:]
    state = np.concatenate([robot_state, gripper_width], axis=1)
    timestamps = zarr_data["timestamp"][:]

    episode_data_index = {
        "from": np.concatenate(([0], zarr_data.meta["episode_ends"][:-1])),
        "to": zarr_data.meta["episode_ends"],
    }

    # Calculate success and reward based on the overlapping area
    # of the T-object and the T-area.
    success = np.zeros_like(action[:, 0], dtype=bool)
    reward = np.zeros_like(action[:, 0], dtype=float)

    # make last frame of each episode a succes, assuming perfect demonstrations
    for i in episode_data_index["to"]:
        success[i - 1] = True
        reward[i - 1] = 1.0

    target_shape = (320, 240)

    features = build_features(mode, n_cameras=n_camera_views, img_shape=(3, target_shape[1], target_shape[0]))
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        features=features,
        image_writer_threads=4,
    )
    episodes = range(len(episode_data_index["from"]))
    if n_episodes:
        episodes = episodes[:n_episodes]
    for ep_idx in tqdm.tqdm(episodes):
        from_idx = episode_data_index["from"][ep_idx]
        to_idx = episode_data_index["to"][ep_idx]
        num_frames = to_idx - from_idx
        print(f"Episode {ep_idx} has {num_frames} steps")
        episode_timestamps = timestamps[from_idx:to_idx]

        # load the videos
        images = {}
        for camera_idx in range(n_camera_views):
            video_path = video_dir / str(ep_idx) / f"{camera_idx}.mp4"
            frames = get_synced_frames_from_mp4_video(video_path, episode_timestamps)

            print(f"Loaded {len(frames)} frames from {video_path}")
            images[f"observation.image.camera_{camera_idx}"] = frames

        # resize all images
        for key in images:
            images[key] = [cv2.resize(img, target_shape) for img in images[key]]

        for frame_idx in range(num_frames):
            i = from_idx + frame_idx
            frame = {
                "action": torch.from_numpy(action[i]),
                # Shift reward and success by +1 until the last item of the episode
                "next.reward": reward[i + (frame_idx < num_frames - 1)],
                "next.success": success[i + (frame_idx < num_frames - 1)],
            }

            frame["observation.state"] = torch.from_numpy(state[i])

            for img_key, frames in images.items():
                frame[img_key] = torch.from_numpy(frames[frame_idx])
            # for key in frame.keys():
            #     print(key, frame[key].shape)
            dataset.add_frame(frame)

        dataset.save_episode(task=TASK_DESCRIPTION)

    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    # To try this script, modify the repo id with your own HuggingFace user (e.g cadene/pusht)
    repo_id = "tlpss/pick-cb-dp-v1"

    raw_dir = Path("/home/tlips/Code/diffusion_policy/data/demo_pick_cb-v2/")

    main(raw_dir=raw_dir, repo_id=repo_id, mode="video", push_to_hub=True, n_episodes=None)
