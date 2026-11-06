"""
Export a trained PPO policy's weights to a C header for on-device inference.

    python export_model.py --model models/walk/final_model.zip
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO


def export_to_c_header(model_path, output_path="firmware_walk/main/model_weights.h"):
    model = PPO.load(model_path)
    policy = model.policy

    layers = []
    for name, param in policy.mlp_extractor.policy_net.named_parameters():
        layers.append((f"policy_{name.replace('.', '_')}", param.detach().cpu().numpy()))
    for name, param in policy.action_net.named_parameters():
        layers.append((f"action_{name.replace('.', '_')}", param.detach().cpu().numpy()))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write("// Auto-generated model weights — do not edit\n")
        f.write(f"// Exported from: {model_path}\n")
        f.write("#pragma once\n\n")

        for name, arr in layers:
            flat = arr.flatten()
            f.write(f"// Shape: {arr.shape}\n")
            if arr.ndim == 2:
                f.write(f"const int {name}_rows = {arr.shape[0]};\n")
                f.write(f"const int {name}_cols = {arr.shape[1]};\n")
            else:
                f.write(f"const int {name}_size = {arr.shape[0]};\n")
            f.write(f"const float {name}[] = {{\n")
            for i in range(0, len(flat), 8):
                chunk = flat[i:i+8]
                f.write("  " + ", ".join(f"{v:.8f}f" for v in chunk) + ",\n")
            f.write("};\n\n")

        f.write("// Network architecture: input(16) -> dense(128,tanh) -> dense(128,tanh) -> output(4)\n")
        f.write(f"const int INPUT_DIM = 16;\n")
        f.write(f"const int HIDDEN_DIM = 128;\n")
        f.write(f"const int OUTPUT_DIM = 4;\n")

    print(f"Exported to {output_path}")
    print(f"Architecture: 16 -> 128 (tanh) -> 128 (tanh) -> 4")
    total_params = sum(arr.size for _, arr in layers)
    print(f"Total parameters: {total_params}")
    print(f"Memory: ~{total_params * 4 / 1024:.1f} KB (float32)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/walk/final_model.zip")
    parser.add_argument("--output", type=str, default="firmware_walk/main/model_weights.h")
    args = parser.parse_args()
    export_to_c_header(args.model, args.output)
