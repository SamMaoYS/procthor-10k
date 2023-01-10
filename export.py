import prior
import json
import argparse
import os
from tqdm import tqdm

def save_json(data, output_path, indent=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w+') as f:
        json.dump(data, f, indent=indent)

def export_split(dataset, split, output_dir):
    print(f'Exporting split {split}')
    split_data = dataset[split].data
    num_houses = len(split_data)
    for i in tqdm(range(num_houses)):
        house_data = split_data[i]
        house_dict = json.loads(house_data.decode('utf-8'))
        save_json(house_dict, os.path.join(output_dir, split, f'procthor-{split}-{i}.json'))

def main(args):
    dataset = prior.load_dataset("procthor-10k")
    splits = ['train', 'val', 'test']
    if args.split in splits:
        export_split(dataset, args.split, args.output_dir)
    elif not args.split:
        for split in splits:
            export_split(dataset, split, args.output_dir)
    else:
        raise ValueError(f"Unknown split: {args.split}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=False, default='')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)