import gc
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from glob import glob
from math import ceil
from typing import Literal

import click
import cv2
import mediapy
import numpy as np
import torch
from diffusers.models import AutoencoderKLTemporalDecoder
from joblib import delayed, Parallel
from tqdm import tqdm


@dataclass
class SampleInfo:
    episode_id: int
    frame_ids: list[int]
    actions: np.ndarray[tuple[Literal["1"], Literal["32"]], np.float32]


class ItemProcessor:
    def __init__(
        self,
        latents_dir_path: str,
        svd_path: str,
        device: torch.device,
        vae_batch_size: int,
    ):
        self._latents_dir_path = latents_dir_path
        self._vae_batch_size = vae_batch_size

        self._vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_path,
            subfolder="vae",
        ).to(device)

        self._device = device

    @torch.inference_mode()
    def __call__(
        self,
        episode_info_path: str,
    ) -> int:
        with open(episode_info_path, "r") as f:
            episode_info = json.load(f)

        episode_id = episode_info["episode_id"]
        video_path = episode_info["videos"][0]["video_path"]

        video = mediapy.read_video(video_path)
        frames = ((torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0) * 2 - 1).to(self._device)
        # L, C, H, W

        assert len(frames) == episode_info["video_length"], \
            "check 'mediapy.read_video': it has read less frames " + \
                f"({episode_info['video_length']}) than it was stored ({len(frames)})"

        latents = []

        for i in range(0, len(frames), self._vae_batch_size):
            batch = frames[i:i + self._vae_batch_size]
            latent = self._vae.encode(batch).latent_dist.sample().mul_(self._vae.config.scaling_factor).cpu()

            latents.append(latent)

        x = torch.cat(
            latents,
            dim=0,
        )

        current_episode_latents_file_path = os.path.join(self._latents_dir_path, f"{episode_id}.pt")
        torch.save(
            x,
            current_episode_latents_file_path,
        )

        del x

        current_episode_latent = (
            {
                "latent_video_path": current_episode_latents_file_path,
            },
        )

        episode_info["latent_videos"] = current_episode_latent

        with open(episode_info_path, "w") as f:
            json.dump(episode_info, f)


def get_frames(
    frames_pathes: list[str],
    forced_width: int,
    forced_height: int,    
) -> list[np.ndarray]:
    frames = [cv2.imread(frame_path) for frame_path in frames_pathes]

    for i in range(len(frames)):
        if frames[i].shape != (forced_height, forced_width, 3):
            frames[i] = cv2.resize(
                frames[i],
                (forced_width, forced_height),
                interpolation=cv2.INTER_LANCZOS4,
            )[:, :, ::-1]

    return frames


def save_video(
    frames: list[np.ndarray],
    output_file_path: str,
    fps: int,
) -> None:
    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def process_single_item(
    item_dir_path: str,
    saved_videos_dir_path: str,
    processed_jsons_dir_path: str,
    item_to_split_mapping_json: str,
    forced_width: int,
    forced_height: int,
    fps: int,
) -> None:
    item_id = item_dir_path.split("/")[-1]

    with open(item_to_split_mapping_json, "r") as f:
        item_to_split_mapping = json.load(f)

    data_split = item_to_split_mapping[item_id]

    if data_split == "test":
        return

    processed_jsons_dir_path = os.path.join(
        processed_jsons_dir_path,
        data_split,
    )
    os.makedirs(
        processed_jsons_dir_path,
        exist_ok=True,
    )

    markup_file_path = os.path.join(item_dir_path, f"{item_id}.json")
    assert os.path.isfile(markup_file_path), f"markup for the item {item_dir_path} wasn't found"

    images_dir_path = os.path.join(item_dir_path, "images")
    os.path.isdir(images_dir_path), f"images for the item {item_dir_path} weren't found"

    with open(markup_file_path, "r") as f:
        markup = json.load(f)

    task = markup["steps"][0]["observation"]["natural_language_instruction"]

    images_pathes = glob(os.path.join(images_dir_path, "*.png"))
    images_pathes = sorted(
        images_pathes,
        key=lambda image_path: int(image_path.split("/")[-1][:-4]),
    )

    frames = get_frames(
        images_pathes,
        forced_width,
        forced_height,
    )

    video_file_path = os.path.join(
        saved_videos_dir_path,
        f"{item_id}.mp4",
    )

    save_video(
        frames,
        video_file_path,
        fps,
    )

    actions = []

    for step_index in range(len(markup["steps"])):
        current_action_dict = markup["steps"][step_index]["action"]
        actions.append(
            current_action_dict["rotation_delta"] + \
            current_action_dict["world_vector"] + \
            current_action_dict["gripper_closedness_action"],
        )

    assert len(frames) == len(actions), \
        f"number of frames {len(frames)} != number of actions {len(actions)}"

    description_dct = {
        "tasks": [task],
        "episode_id": item_id,
        "video_length": len(frames),
        "videos": [
            {
                "video_path": video_file_path,
            },
        ],
        "actions": actions,
    }

    with open(
        os.path.join(processed_jsons_dir_path, f"{item_id}.json"),
        "w",
    ) as f:
        json.dump(description_dct, f)


def process_single_chunk(
    episode_pathes: list[str],
    episode_processor: ItemProcessor,
) -> None:
    for episode_path in tqdm(episode_pathes):
        episode_processor(episode_path)

        torch.cuda.empty_cache()
        gc.collect()


def initialize_annotation_files(data_split_dir_path: str) -> list[str]:
    ann_files = [
        os.path.join(data_split_dir_path, f) for f in os.listdir(data_split_dir_path) if f.endswith(".json")
    ]

    return ann_files


def load_and_process_annotation_file(ann_file: str) -> list[SampleInfo]:
    samples = []

    with open(ann_file, "r") as f:
        annotation = json.load(f)

    n_frames = annotation["video_length"]

    for start_frame in range(0, n_frames - 1):
        sample = SampleInfo(
            episode_id=annotation["episode_id"],
            frame_ids=[start_frame],
            actions=np.array(annotation["actions"])[start_frame:start_frame + 1],
        )
        samples.append(sample)

    return samples


def init_sequences(ann_files: list[str]) -> list[SampleInfo]:
    samples = []

    samples = Parallel(n_jobs=-1)(
        delayed(load_and_process_annotation_file)(
            ann_file,
        ) for ann_file in ann_files
    )

    samples = [item for sample in samples for item in sample]

    return samples


def extract_meta_info(
    processed_dataset_dir_path: str,
    meta_info_dir_path: str,
):
    for data_type in ["train", "val"]:
        annotation_files = initialize_annotation_files(os.path.join(processed_dataset_dir_path, data_type))

        logging.info(f"for the {data_type} part {len(annotation_files)} annotation files were found")

        samples_all = init_sequences(annotation_files)

        logging.info(f"{len(samples_all)} samples were found")

        if data_type == "train":
            actions_all = np.stack(
                [sample.actions.squeeze(0) for sample in samples_all],
                axis=0,
            )

            logging.info(f"{actions_all.shape} shape of all actions")  # N, 7

            actions_01 = np.percentile(
                actions_all,
                1,
                axis=0,
            )
            actions_99 = np.percentile(
                actions_all,
                99,
                axis=0,
            )

            stat = {
                "actions_01": actions_01.tolist(),
                "actions_99": actions_99.tolist(),
            }

            with open(
                os.path.join(meta_info_dir_path, "stat.json"),
                "w",
            ) as f:            
                json.dump(stat, f)

        samples_all_as_dict = [asdict(sample) for sample in samples_all]
        for samples in samples_all_as_dict:
            del samples["actions"]

        if data_type == "train":
            random.shuffle(samples_all)

        with open(os.path.join(meta_info_dir_path, f"{data_type}_sample.json"), "w") as f:
            json.dump(samples_all_as_dict, f)


@click.command()
@click.option("--dataset_dir_path", type=click.Path(exists=True))
@click.option("--saved_videos_dir_path", type=click.Path())
@click.option("--processed_jsons_dir_path", type=click.Path())
@click.option("--latents_dir_path", type=click.Path())
@click.option("--meta_info_dir_path", type=click.Path())
@click.option("--item_to_split_mapping_json", type=click.Path(exists=True))
@click.option("--svd_dir_path", type=click.Path(exists=True))
@click.option("--forced_height", type=int, default=256)
@click.option("--forced_width", type=int, default=320)
def main(
    dataset_dir_path: str,
    saved_videos_dir_path: str,
    processed_jsons_dir_path: str,
    latents_dir_path: str,
    meta_info_dir_path: str,
    item_to_split_mapping_json: str,
    svd_dir_path: str,
    forced_height: int = 256,
    forced_width: int = 320,
) -> None:
    os.makedirs(
        saved_videos_dir_path,
        exist_ok=False,
    )

    os.makedirs(
        processed_jsons_dir_path,
        exist_ok=False,
    )

    os.makedirs(
        latents_dir_path,
        exist_ok=False,
    )

    os.makedirs(
        meta_info_dir_path,
        exist_ok=False,
    )

    items_dirs = glob(os.path.join(dataset_dir_path, "**"))

    logging.info(f"{len(items_dirs)} items will be processed")

    Parallel(n_jobs=-1)(
        delayed(process_single_item)(
            item_dir_path,
            saved_videos_dir_path,
            processed_jsons_dir_path,
            item_to_split_mapping_json,
            forced_width,
            forced_height,
            5,
        ) for item_dir_path in tqdm(items_dirs)
    )

    logging.info("main processing stage has ended")

    logging.info("latents extraction has started")

    num_gpus = torch.cuda.device_count()
    logging.info(f"{num_gpus} devices were found")

    episodes_pathes = glob(os.path.join(processed_jsons_dir_path, "**", "*.json"))
    logging.info(f"{len(episodes_pathes)} processed items were found")

    n_episodes_per_chunk = ceil(len(episodes_pathes) / num_gpus)

    chunks = [
        episodes_pathes[chunk_index * n_episodes_per_chunk: (chunk_index + 1) * n_episodes_per_chunk] for \
        chunk_index in range(num_gpus)
    ]

    logging.info(f"items were splitted as {' '.join([str(len(c)) for c in chunks])}")

    assert sum([len(chunk) for chunk in chunks]) == len(episodes_pathes), \
        "sum of episodes in chunks must be equal to the original number of episodes"

    item_processors = []
    for i in range(num_gpus):
        item_processors.append(
            ItemProcessor(
                latents_dir_path,
                svd_dir_path,
                torch.device(f"cuda:{i}"),
                128,
            )
        )

    Parallel(n_jobs=num_gpus)(
        delayed(process_single_chunk)(
            chunk,
            item_processor,
        ) for chunk, item_processor in zip(chunks, item_processors)
    )

    logging.info("latents were added")

    logging.info("extracting meta information")

    extract_meta_info(
        processed_jsons_dir_path,
        meta_info_dir_path,
    )

    logging.info("meta information was extracted")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()
