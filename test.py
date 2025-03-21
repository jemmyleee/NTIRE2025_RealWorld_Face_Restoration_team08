import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        # CodeFormer baseline, NIPS 2022
        from models.team00_CodeFormer import main as CodeFormer
        name = f"{model_id:02}_CodeFormer_baseline"
        model_path = os.path.join('model_zoo', 'team00_CodeFormer')
        model_func = CodeFormer
    elif model_id == 8:
        from models.team08_good import main as good
        name = f"{model_id:02}_good"
        model_path = os.path.join('model_zoo', 'team08_good')
        model_func = good
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    return model_func, model_path, name


def run(model_func, model_name, model_path, device, args, mode="test"):
    # --------------------------------
    # dataset path
    # --------------------------------
    if mode == "valid":
        data_path = args.valid_dir
    elif mode == "test":
        data_path = args.test_dir
    assert data_path is not None, "Please specify the dataset path for validation or test."
    
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    data_paths = []
    save_paths = []
    for dataset_name in ("CelebA", "Wider-Test", "LFW-Test", "WebPhoto-Test", "CelebChild-Test"):
        data_paths.append(os.path.join(data_path, dataset_name))
        save_paths.append(os.path.join(save_path, dataset_name))
        util.mkdir(save_paths[-1])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for data_path, save_path in zip(data_paths, save_paths):
        model_func(model_dir=model_path, input_path=data_path, output_path=save_path, device=device,args=args)
    end.record()
    torch.cuda.synchronize()
    print(f"Model {model_name} runtime (Including I/O): {start.elapsed_time(end)} ms")


def main(args):

    utils_logger.logger_info("NTIRE2025-RealWorld-Face-Restoration", log_path="NTIRE2025-RealWorld-Face-Restoration.log")
    logger = logging.getLogger("NTIRE2025-RealWorld-Face-Restoration")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model_func, model_path, model_name = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if args.valid_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="valid")
        
    if args.test_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="test")
# =========================================================================================================
# =========================================================================================================
if __name__ == "__main__":
    DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
    "skin pore detailing, hyper sharpness, perfect without deformations."
)
    parser = argparse.ArgumentParser("NTIRE2025-RealWorld-Face-Restoration")
    parser.add_argument("--valid_dir", default=None, type=str, help="Path to the validation set")
    parser.add_argument("--input", default=None, type=str, help="input")
    parser.add_argument("--test_dir", default=None, type=str, help="Path to the test set")
    parser.add_argument("--save_dir", default="NTIRE2025-RealWorld-Face-Restoration/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument(
        "--output", type=str, help="Path to save restored results."
    )
    parser.add_argument(
    "--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--task",
        type=str,
        default="face",
        choices=["sr", "face", "denoise", "unaligned_face"],
        help="Task you want to do. Ignore this option if you are using self-trained model.",
    )
    parser.add_argument(
        "--upscale", type=float, default=1, help="Upscale factor of output."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2", "v2.1", "custom"],
        help="DiffBIR model version.",
    )
    parser.add_argument(
        "--train_cfg",
        type=str,
        default="",
        help="Path to training config. Only works when version is custom.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Path to saved checkpoint. Only works when version is custom.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="spaced",
        # choices=[
        #     "dpm++_m2",
        #     "spaced",
        #     "ddim",
        #     "edm_euler",
        #     "edm_euler_a",
        #     "edm_heun",
        #     "edm_dpm_2",
        #     "edm_dpm_2_a",
        #     "edm_lms",
        #     "edm_dpm++_2s_a",
        #     "edm_dpm++_sde",
        #     "edm_dpm++_2m",
        #     "edm_dpm++_2m_sde",
        #     "edm_dpm++_3m_sde",
        # ],
        help="Sampler type. Different samplers may produce very different samples.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=75,
        help="Sampling steps. More steps, more details.",
    )
    parser.add_argument(
        "--start_point_type",
        type=str,
        choices=["noise", "cond"],
        default="noise",
        help=(
            "For DiffBIR v1 and v2, setting the start point types to 'cond' can make the results much more stable "
            "and ensure that the outcomes from ODE samplers like DDIM and DPMS are normal. "
            "However, this adjustment may lead to a decrease in sample quality."
        ),
    )
    parser.add_argument(
        "--cleaner_tiled",
        action="store_true",
        help="Enable tiled inference for stage-1 model, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--cleaner_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(
        "--cleaner_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(
        "--vae_encoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE encoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_encoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--vae_decoder_tiled",
        action="store_true",
        help="Enable tiled inference for AE decoder, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--vae_decoder_tile_size", type=int, default=256, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tiled",
        action="store_true",
        help="Enable tiled sampling, which reduces the GPU memory usage.",
    )
    parser.add_argument(
        "--cldm_tile_size", type=int, default=512, help="Size of each tile."
    )
    parser.add_argument(
        "--cldm_tile_stride", type=int, default=256, help="Stride between tiles."
    )
    parser.add_argument(
        "--captioner",
        type=str,
        choices=["none", "llava", "ram"],
        default="none",
        help="Select a model to describe the content of your input image.",
    )
    parser.add_argument(
        "--pos_prompt",
        type=str,
        default=DEFAULT_POS_PROMPT,
        help=(
            "Descriptive words for 'good image quality'. "
            "It can also describe the things you WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default='low quality, blurry, low-resolution, noisy, unsharp, weird textures',
        help=(
            "Descriptive words for 'bad image quality'. "
            "It can also describe the things you DON'T WANT to appear in the image."
        ),
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--rescale_cfg",
        action="store_true",
        help="Gradually increase cfg scale from 1 to ('cfg_scale' + 1)",
    )
    parser.add_argument(
        "--noise_aug",
        type=int,
        default=0,
        help="Level of noise augmentation. More noise, more creative.",
    )
    parser.add_argument(
        "--s_churn",
        type=float,
        default=0,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmin",
        type=float,
        default=0,
        help="Minimum sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_tmax",
        type=float,
        default=300,
        help="Maximum  sigma for adding ramdomness to sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--s_noise",
        type=float,
        default=1,
        help="Randomness in sampling. Only works with some edm samplers.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1,
        help="I don't understand this parameter. Leave it as default.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Order of solver. Only works with edm_lms sampler.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="Control strength from ControlNet. Less strength, more creative.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Nothing to say.")
    # guidance parameters
    parser.add_argument(
        "--guidance", action="store_true", help="Enable restoration guidance."
    )
    parser.add_argument(
        "--g_loss",
        type=str,
        default="w_mse",
        choices=["mse", "w_mse"],
        help="Loss function of restoration guidance.",
    )
    parser.add_argument(
        "--g_scale",
        type=float,
        default=0.0,
        help="Learning rate of optimizing the guidance loss function.",
    )
    # common parameters

    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples for each image."
    )
    parser.add_argument("--seed", type=int, default=231)
    # mps has not been tested

    parser.add_argument(
        "--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--llava_bit", type=str, default="4", choices=["16", "8", "4"])
    args = parser.parse_args()
    pprint(args)

    main(args)
