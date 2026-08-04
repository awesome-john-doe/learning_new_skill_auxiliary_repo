import json
from glob import glob
from pathlib import Path
import os

import click
import numpy as np
from joblib import delayed, Parallel
from loguru import logger
from PIL import Image
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio


def _sorted_image_paths(path_to_images_glob: str) -> list[str]:
    paths = glob(path_to_images_glob)

    try:
        return sorted(paths, key=lambda p: int(Path(p).stem))
    except Exception:
        return sorted(paths)

def _load_and_resize_rgb(
    image_path: str | Path,
    forced_height: int,
    forced_width: int,
) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")

    if img.size != (forced_width, forced_height):
        img = img.resize((forced_width, forced_height), Image.LANCZOS)

    return np.asarray(img, dtype=np.uint8)

def convert_single_item(
    path_to_item_dir: str | Path,
    single_chunk_size: int,
    step_size: int,
    item_to_split_mapping: dict[str, str], 
    path_to_output_dir: str | Path,
    forced_height: int,
    forced_width: int,
) -> None:
    item_index = str(path_to_item_dir).split(os.path.sep)[-1]

    logger.info(f"now converting item #{item_index}")

    split = item_to_split_mapping[item_index]

    path_to_item_dir = Path(path_to_item_dir)

    path_to_output_dir = Path(path_to_output_dir) / split
    path_to_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(glob(str(path_to_item_dir / "*.json"))[-1], "r", encoding="utf-8") as fractal_style_json:
        fractal_style_json = json.load(fractal_style_json)

    actions = [step["action"] for step in fractal_style_json["steps"]]

    path_to_images = _sorted_image_paths(str(path_to_item_dir / "images" / "*.png"))

    n_images = len(path_to_images)

    if n_images < single_chunk_size:
        logger.info(f"padding item_index #{item_index} to the minimal length of {single_chunk_size} frames")

        diff = single_chunk_size - n_images

        path_to_images += [path_to_images[-1]] * diff

        zero_action = {
            "rotation_delta": [0.0, 0.0, 0.0],
            "world_vector": [0.0, 0.0, 0.0],
            "gripper_closedness_action": [0.0],
        }
        actions += [zero_action] * diff

        assert len(path_to_images) == single_chunk_size

    logger.info(f"found {len(path_to_images)} steps for the item #{item_index}")

    for chunk_index, chunk_start in enumerate(range(0, len(actions), step_size)):
        chunk_actions = actions[chunk_start:chunk_start + single_chunk_size]
        chunk_image_paths = path_to_images[chunk_start:chunk_start + single_chunk_size]

        if len(chunk_actions) != single_chunk_size:
            break

        # Prepare frames
        frames_rgb: list[np.ndarray] = [
            _load_and_resize_rgb(image_path, forced_height, forced_width)
            for image_path in chunk_image_paths
        ]

        # Write video
        video_path = path_to_output_dir / f"{chunk_index}_{item_index}.mp4"
        with imageio.get_writer(str(video_path), fps=1, macro_block_size=1) as writer:
            for frame in frames_rgb:
                writer.append_data(frame)

        # Write actions JSON as {frame_num: {"actions": one_hot_vector}}
        actions_dict: dict[str, dict[str, list[int]]] = {}

        for content_index, action in enumerate(chunk_actions):
            actions_dict[str(content_index)] = {
                "mouse": [0, 0],
                "keyboard": action["rotation_delta"] + action["world_vector"] + action["gripper_closedness_action"],
            }

            assert len(actions_dict[str(content_index)]["keyboard"]) == 7

        actions_json_path = path_to_output_dir / f"{chunk_index}_{item_index}.json"

        with open(actions_json_path, "w", encoding="utf-8") as f:
            json.dump(actions_dict, f)

    if len(actions) % (single_chunk_size - step_size) != 0 and len(actions) % (single_chunk_size) !=0:
        chunk_index = len(actions) // (single_chunk_size - step_size)

        logger.info(f"creating additional last chunk for the item #{item_index} with chunk index {chunk_index}")

        chunk_actions = actions[-single_chunk_size:]
        chunk_image_paths = path_to_images[-single_chunk_size:]

        frames_rgb: list[np.ndarray] = [
            _load_and_resize_rgb(image_path, forced_height, forced_width)
            for image_path in chunk_image_paths
        ]

        video_path = path_to_output_dir / f"-1_{item_index}.mp4"

        with imageio.get_writer(str(video_path), fps=1, macro_block_size=1) as writer:
            for frame in frames_rgb:
                writer.append_data(frame)

        actions_dict: dict[str, dict[str, list[int]]] = {}

        for content_index, action in enumerate(chunk_actions):
            actions_dict[str(content_index)] = {
                "mouse": [0, 0],
                "keyboard": action["rotation_delta"] + action["world_vector"] + action["gripper_closedness_action"],
            }

            assert len(actions_dict[str(content_index)]["keyboard"]) == 7

        actions_json_path = path_to_output_dir / f"-1_{item_index}.json"

        with open(actions_json_path, "w", encoding="utf-8") as f:
            json.dump(actions_dict, f)

        logger.success(f"{chunk_index + 1} video/json files created for the item #{item_index}")
    else:
        logger.success(f"{chunk_index} video/json files created for the item #{item_index}")


@click.command()
@click.option("--items_dir_path", type=click.Path(exists=True))
@click.option("--output_dir_path", type=click.Path())
@click.option("--item_to_split_mapping_file_path", type=click.Path(exists=True))
@click.option("--chunk_size", type=int, default=57)
@click.option("--step_size", type=int, default=19)
@click.option("--forced_height", type=int, default=256)
@click.option("--forced_width", type=int, default=320)
def main(
    items_dir_path: str,
    output_dir_path: str,
    item_to_split_mapping_file_path: str,
    chunk_size: int,
    step_size: int,
    forced_height: int,
    forced_width: int,
) -> None:
    items_pathes = glob(str(Path(items_dir_path) / "*"))

    logger.info(f"{len(items_pathes)} items were found")

    with open(item_to_split_mapping_file_path, "r") as f:
        item_to_split_mapping = json.load(f)

    logger.info("starting dataset conversion to videos/jsons")

    Parallel(n_jobs=-1)(
        delayed(convert_single_item)(
            path_to_item_dir,
            chunk_size,
            step_size,
            item_to_split_mapping,
            output_dir_path,
            forced_height,
            forced_width,
        ) for path_to_item_dir in tqdm(items_pathes)
    )

    logger.info("dataset was converted")


if __name__ == "__main__":
    main()
