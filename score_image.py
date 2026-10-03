import argparse
import json

import cv2
import torch

from model.IQA_Model import IQA_Model
from patch_sampling.Patch_Sampler import SaliencyPatchSampler

CONFIG_PATH = "options/main_training/koniq.json"


@torch.inference_mode()
def main(checkpoint_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CONFIG_PATH) as f:
        args = {k: v for group in json.load(f).values() for k, v in group.items()}

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model = IQA_Model(device=device, options=args).to(device)
    sampler = SaliencyPatchSampler(
        patch_size=args["patch_size"],
        num_scales=args["num_scales"],
        min_short_edge=args["min_short_edge"],
        device=device,
    ).to(device)

    model.load_state_dict(state["model_ema"])
    sampler.load_state_dict(state["patch_sampler_ema"])
    model.eval()
    sampler.eval()

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).to(
        device=device, dtype=torch.float32
    ) / 255.0

    patches, weights, scale_counts = sampler(
        image,
        num_patches=args["train_num_patches"],
        stride_ratio=args["patch_stride"],
        random=False,
    )
    score = model(patches.unsqueeze(0), weights.unsqueeze(0), scale_counts)
    print(f"Score: {score.item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="model_weights.pth")
    parser.add_argument("--image_path", default="image.jpg")
    main(**vars(parser.parse_args()))