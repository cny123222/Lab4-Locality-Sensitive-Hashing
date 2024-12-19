import sys
import glob
import time
import argparse
from lsh import LSH
from knn import KNN


def parse_args(args):
    """
    读取命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="dataset", help="Path to folder containing dataset images")
    parser.add_argument("--type", type=str, choices=["LSH", "KNN", "comp"], default="LSH", help="Choose the algorithm to use")
    parser.add_argument("--indicator", type=list, default=[1, 5, 9, 13, 17, 21], help="Projection indicators for LSH")
    parser.add_argument("--target-dir", type=str, default="target.jpg", help="path to the target image")

    args = parser.parse_args(args)
    return args


def get_image_paths(input_dir, extensions = ("jpg", "jpeg", "png", "bmp")):
    """
    找到所有图片文件路径
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

    tasks = ["LSH", "KNN"] if args.type == "comp" else [args.type]

    for type in tasks:
        searcher = LSH(args.indicator) if type == "LSH" else KNN()
        img_paths = get_image_paths(args.image_dir)
        for img_path in img_paths:
            searcher.add(img_path)

        start_time = time.time()
        result_path = searcher.search(args.target_dir)
        finish_time = time.time()

        if args.type == "comp":
            if type == "LSH":
                lsh_time = finish_time - start_time
            else:
                knn_time = finish_time - start_time

        print()
        print(f"{type}:")
        if type == "LSH":
            print(f"Indicator:{args.indicator}")
        print("Most similar image:", result_path)
        print("Time taken:", finish_time - start_time)

    if args.type == "comp":
        speed_up = knn_time / lsh_time
        print()
        print(f"Speed up: {speed_up:.2f}x")


if __name__ == '__main__':
    main(sys.argv[1:])