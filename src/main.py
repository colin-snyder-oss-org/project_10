# src/main.py
import argparse
from src.config import get_config
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description='Hierarchical VAE for Anomaly Detection')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'inference'],
                        help='Mode to run the script in')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    config = get_config(args.config)

    if args.mode == 'train':
        trainer = Trainer(config)
        trainer.train()
    elif args.mode == 'evaluate':
        evaluator = Evaluator(config)
        evaluator.evaluate()
    elif args.mode == 'inference':
        evaluator = Evaluator(config)
        evaluator.run_inference()
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
