import argparse, yaml, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--dataset', help='Path to dataset or JSON with GT/paths (TODO)')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    print('Evaluation harness stub. Fill in dataset loader and per-model runs.')

if __name__ == '__main__':
    main()
