import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--name", type=str, default="cora_40")
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument("--emblem_size", default=512, type=int)
    parser.add_argument("--max_epoch", type=int, default=100) 
    parser.add_argument("--lambda1", type=float, default=1e+1)
    parser.add_argument("--lambda2", type=float, default=1e+1)
    parser.add_argument("--lambda3", type=float, default=1e+1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=int, default=5e-5)
    args = parser.parse_args()
    
    train(args)
