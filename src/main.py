import sys
import glob
import time
import argparse
from src.lsh import LSH
from src.knn import KNN


def parse_args(args):
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="dataset", help="Path to folder containing dataset images")
    parser.add_argument("--target-dir", type=str, default="target.jpg", help="Path to the target image")
    parser.add_argument("--type", type=str, choices=["LSH", "NN", "comp"], default="LSH", help="Choose the algorithm to use, comp for comparison")
    parser.add_argument("--indicator", type=int, nargs='+', default=[1, 8, 16, 24], help="Projection set for LSH")
    parser.add_argument("--resnet", type=bool, default=False, help="Use ResNet to generate feature vector")

    args = parser.parse_args(args)
    return args


def get_image_paths(input_dir, extensions = ("jpg", "jpeg", "png", "bmp")):
    """
    Get image paths from the given directory
    """
    pattern = f"{input_dir}/**/*"
    img_paths = []
    for extension in extensions:
        img_paths.extend(glob.glob(f"{pattern}.{extension}", recursive=True))

    if not img_paths:
        raise FileNotFoundError(f"No images found in {input_dir}. Supported formats are: {', '.join(extensions)}")

    return img_paths


def main(args):
    args = parse_args(args)

    print(args.indicator)

    tasks = ["LSH", "NN"] if args.type == "comp" else [args.type]

    for type in tasks:
        # Initialize the searcher
        searcher = LSH(args.indicator, args.resnet) if type == "LSH" else KNN(args.resnet)

        # Add images to the searcher
        img_paths = get_image_paths(args.image_dir)
        for img_path in img_paths:
            searcher.add(img_path)

        # Search for the most similar image
        start_time = time.time()
        result_path = searcher.search(args.target_dir)
        finish_time = time.time()

        # Record time taken for each algorithm
        if args.type == "comp":
            if type == "LSH":
                lsh_time = finish_time - start_time
            else:
                knn_time = finish_time - start_time

        print()
        print(f"{type}:")
        if type == "LSH":
            print(f"Projection Set:{args.indicator}")
        print(f"Most similar image: {result_path}")
        print(f"Time taken: {finish_time - start_time}s")

    # Compare the time taken for LSH and NN
    if args.type == "comp":
        speed_up = knn_time / lsh_time
        print()
        print(f"Speed up: {speed_up:.2f}x")


if __name__ == '__main__':
    main(sys.argv[1:])