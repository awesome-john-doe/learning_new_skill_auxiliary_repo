import json
from math import ceil

import click
from loguru import logger


@click.command()
@click.option("--instruction_to_items_mapping_file_path", type=click.Path(exists=True))
@click.option("--train_val_test_ratio", type=str)
@click.option("--item_to_split_file_path", type=click.Path())
def main(
    instruction_to_items_mapping_file_path: str,
    train_val_test_ratio: str,
    item_to_split_file_path: str,
) -> None:
    assert train_val_test_ratio.count("/") == 2, "'train_val_test_ratio' must separates 3 ints with 2 slashes"

    train_part, val_part, test_part = train_val_test_ratio.split("/")

    assert train_part.isdigit(), "train part must be an int"
    assert val_part.isdigit(), "val part must be an int"
    assert test_part.isdigit(), "test part must be an int"

    train_part = int(train_part)
    val_part = int(val_part)
    test_part = int(test_part)

    total_sum = train_part + val_part + test_part
    assert total_sum == 100, f"sum of train, val, and test parts must be 100; now sum is {total_sum}"

    with open(instruction_to_items_mapping_file_path, "r") as f:
        instruction_to_items_mapping = json.load(f)

    logger.info("train-val-test mapping creating has started")

    item_to_split_mapping = {}

    for instruction, items in instruction_to_items_mapping.items():
        n_items = len(items)

        n_train_items = ceil((train_part / 100) * n_items)
        n_val_items = 0 if n_train_items == n_items else ceil((val_part / 100) * n_items)
        n_test_items = n_items - n_train_items - n_val_items

        train_items = items[:n_train_items]
        val_items = items[n_train_items:n_train_items+n_val_items]
        test_items = items[n_train_items+n_val_items:]

        for items, split_size, split_label in zip(
            [train_items, val_items, test_items],
            [n_train_items, n_val_items, n_test_items],
            ["train", "val", "test"],
        ):
            assert len(items) == split_size, \
                f"check {split_label} part's size for the '{instruction}': {len(items)} != {split_size}"

        for items, split_label in zip(
            [train_items, val_items, test_items],
            ["train", "val", "test"],
        ):
            for item in items:
                item_to_split_mapping[item] = split_label

    with open(item_to_split_file_path, "w") as f:
        json.dump(item_to_split_mapping, f)

    logger.success(f"mapping was saved as {item_to_split_file_path}")


if __name__ == "__main__":
    main()
