# This is an official auxiliary repository for the paper "Learning New Robot Skills from Video Demonstrations with Inverse Dynamics and World Models"

## Data Preprocessing Pipeline

### Dependencies

We've used UV to manage dependencies. Using it, it's easy to download all the packages as:

```bash
uv sync
```

Additional dependencies, for example, for converting data to the required format can be installed as:

```bash
uv sync --extra *group-name*
```

For the Matrix-Game-2.0, the \*group-name\* is matrix, and for the Ctrl-World it's called \*ctrl_world\*.

### Train/Val/Test split

First of all, you need to obtain the data from HuggingFace:

```bash
hf download awesome-john-doe/fractal ./raw.tar.gz --repo-type dataset --local-dir ./data
```

Now directory ./data/fractal contains Fractal (RT-1) dataset.

We've used a stratified train/val/test split, so you need to obtain a mapping from instruction to relevant items:

```bash
python ./scripts/get_instruction_to_items_mapping.py \
    --data_dir_path ./data/fractal \
    --mapping_file_path ./instruction-to-items.json
```

Now a stratified split can be performed:

```bash
python ./scripts/get_train_val_test_split_mapping.py \
    --instruction_to_items_mapping_file_path ./instruction-to-items.json \
    --train_val_test_ratio 80/10/10 \
    --item_to_split_file_path ./item-to-split.json
```

### Matrix-Game-2.0 format

Dataset can be converted to the Matrix-Game-2.0 format as:

Let's download a base Matrix-Game-2.0 checkpoint:

```bash
hf download Skywork/Matrix-Game-2.0 --repo-type model --local-dir ./Matrix-Game-2.0
```

```bash
python ./scripts/convert_to_matrix_format.py \
    --items_dir_path ./data/fractal \
    --output_dir_path ./data/matrix_format \
    --item_to_split_mapping_file_path ./item-to-split.json \
    --chunk_size 57 \
    --step_size 19 \
    --forced_height 256 \
    --forced_width 320
```

### Ctrl-World format

The same thing can be done for the Ctrl-World, but you need to download SVD firstly and CLIP firstly:

```bash
hf download stabilityai/stable-video-diffusion-img2vid --repo-type model --local-dir ./SVD
hf download openai/clip-vit-base-patch32 --repo-type model --local-dir ./CLIP
```

Then the convertion can be performed:

```bash
python ./scripts/convert_to_ctrl_world_format.py \
    --dataset_dir_path ./data/fractal \
    --saved_videos_dir_path ./data/ctrl-world/videos \
    --processed_jsons_dir_path ./data/ctrl-world/processed_jsons \
    --latents_dir_path ./data/ctrl-world/latents \
    --meta_info_dir_path ./data/ctrl-world/meta_info \
    --item_to_split_mapping_json ./item-to-split.json \
    --svd_dir_path ./SVD \
    --forced_height 256 \
    --forced_width 320
```

## Reproducibility

For the [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World), its own train loop was used. In the case of [Matrix-Game-2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2), [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) framework was employed. For the IDM training [NVIDIA GR00T](https://github.com/nvidia/gr00t-dreams) pipeline was used.

All the confings with hypeparameters used in the paper can be found in the [configs](./configs) directory.

## Training details

All the models were trained using single node with 8 H100 GPUs on board with DDP parallelization approach. The approximate training time to convergence is given in the table below.

| Model                   | Trainig time, hours | Number of optimization steps | Number of GPUs |
| ----------------------- | ------------------- | ---------------------------- | -------------- |                                
| Matrix-Game 2.0         | 216                 | 45k                          | 8              |
| Ctrl-World              | 99                  | 45k                          | 8              |
| IDM, all commands       | 197                 | 100k                         | 1              |
| IDM, w/o knock          | 301                 | 100k                         | 1              |
| IDM, w/o open and close | 280                 | 100k                         | 1              |

## Supplementary materials

Detailed human evaluation protocol [report](./reports/human-evaluation-protocol.pdf) is also presented in the repository.
