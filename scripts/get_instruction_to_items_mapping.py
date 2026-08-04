import json
import os
from collections import defaultdict
from dataclasses import dataclass
from glob import glob

import click
from joblib import delayed, Parallel
from loguru import logger
from tqdm import tqdm


@dataclass
class InstructionItemPair:
    instruction: str
    item_id: str


def process_single_file(
    item_dir_path: str,
) -> InstructionItemPair:
    with open(glob(os.path.join(f"{item_dir_path}", "*.json"))[0]) as f:
        fractal_format_json = json.load(f)

    instruction = fractal_format_json["steps"][0]["observation"]["natural_language_instruction"].strip()
    item_id = item_dir_path.split(os.sep)[-1]

    return InstructionItemPair(instruction, item_id)


@click.command()
@click.option("--data_dir_path", type=click.Path(exists=True))
@click.option("--mapping_file_path", type=click.Path())
def main(
    data_dir_path: str,
    mapping_file_path: str,
):
    items_dirs_pathes = glob(os.path.join(f"{data_dir_path}", "**"))

    logger.info(f"{len(items_dirs_pathes)} items were found")

    instruction_item_pairs = Parallel(n_jobs=-1)(
        delayed(process_single_file)(item_dir_path) for item_dir_path in tqdm(items_dirs_pathes)
    )

    logger.info("items processing ended")

    instruction_to_items_mapping = defaultdict(list)
    for pair in instruction_item_pairs:
        instruction_to_items_mapping[pair.instruction].append(pair.item_id)

    instruction_to_items_mapping = {
        k: v for k, v in sorted(
            instruction_to_items_mapping.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )
    }

    message = f"mapping was created; {len(list(instruction_to_items_mapping.keys()))} instructions were found;\n" + \
              f"for every instruction were found:\n"
    
    for instruction in instruction_to_items_mapping:
        message += f"{instruction}: {len(instruction_to_items_mapping[instruction])} examples\n"

    logger.info(message)

    with open(mapping_file_path, "w") as f:
        json.dump(
            instruction_to_items_mapping,
            f,
        )

    logger.success(f"mapping was saved as {mapping_file_path}")


if __name__ == "__main__":
    main()
