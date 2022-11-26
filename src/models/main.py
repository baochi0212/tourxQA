import argparse
import os
import sys

working_dir = os.environ['source'] 

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--n_epochs", default=30, type=int)
parser.add_argument("--pretrained", action="store_true", default=False)
parser.add_argument("--pretrained_model", default="vinai/phobert-base", type=str)
parser.add_argument("--model_path", default=f"{dir}/models/weights/model.pt", type=str)

if __name__ == "__main__":
    args = parser.parse_args()