import torch
from dataclasses import dataclass


@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths
    svd_model_path = "./svd"
    clip_model_path = "./clip"
    ckpt_path = None
    pi_ckpt = None

    # meta info
    dataset_meta_info_path = "./data/ctrl-world/meta_info"
    dataset_path = "./data/ctrl-world/processed_jsons"
    num_workers = 8
    skip_step = 1

    output_dir = "./ctrl-world-fractal-finetune"

    # training parameters
    learning_rate = 1e-5
    gradient_accumulation_steps = 8
    mixed_precision = "fp16"
    train_batch_size = 8
    shuffle = True
    num_train_epochs = 20
    warmup_steps = 5000
    checkpointing_steps = 2500
    validation_steps = 2500
    loss_logging_steps = 100
    max_grad_norm = 1.0
    # for val
    video_num = 10

    ############################ model args ##############################

    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 2
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 256
    num_frames = 5
    num_history = 7
    action_dim = 7
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.float16
