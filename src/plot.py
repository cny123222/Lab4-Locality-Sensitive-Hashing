import sys
import time
import threading
from tqdm import tqdm
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from src.lsh import LSH
from src.knn import KNN
from src.preprocess import Image

# Plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11


def compute_knn_time(
        repeat_times: int = 100
):
    """
    Compute the average time for KNN search
    """
    knn = KNN(normalize=False)  # Skip RGB normalization for better result
    for i in range(1, 41):
        img_path = f"dataset/{i}.jpg"
        knn.add(img_path)

    # Compute the time for KNN search
    knn_time = 0
    for i in range(repeat_times):
        start_time = time.time()
        knn.search("target.jpg")
        knn_time += (time.time() - start_time)
    knn_time /= repeat_times

    return knn_time


def compute_lsh_time(
        indicators: list, 
        repeat_times: Tuple[int, int] = (1, 100),
        multithread: bool = True
):
    """
    Compute the average time for LSH search
    """
    # Compute the time for LSH initialization
    init_time = 0
    for i in range(repeat_times[0]):
        start_time = time.time()
        lsh = LSH(indicators, normalize=False)  # Skip RGB normalization for better result
        if multithread:  # Use multithreading for better performance
            threads = []
            for i in range(1, 41):
                img_path = f"dataset/{i}.jpg"
                thread = threading.Thread(target=lsh.add, args=(img_path,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for i in range(1, 41):
                img_path = f"dataset/{i}.jpg"
                lsh.add(img_path)
        init_time += (time.time() - start_time)
    init_time /= repeat_times[0]

    # Compute the time for LSH search
    search_time = 0
    for i in range(repeat_times[1]):
        start_time = time.time()
        lsh.search("target.jpg")
        search_time += (time.time() - start_time)
    search_time /= repeat_times[1]

    return init_time, search_time


def generate_scatter_indicator(
        bucket_num: int
):
    """
    Generate scatter indicators for LSH
    """
    assert 1 <= bucket_num <= 24
    indicators = []
    indicators = np.linspace(1, 24, bucket_num, dtype=int)
    return indicators


def plot_init_search(
        bucket_nums: range = range(1, 25), 
        repeat_times: Tuple[int, int] = (500, 100), 
        save_path: str = "results/time_init_search.png"
):
    """
    Compare the initialization time and search time of different indicators
    """
    # Compute the time for LSH initialization and search
    lsh_init_times = []
    lsh_search_times = []
    for bucket_num in tqdm(bucket_nums, desc="Different Dimension of Projection Set"):
        indicators = generate_scatter_indicator(bucket_num)
        init_time, search_time = compute_lsh_time(indicators, repeat_times)
        lsh_init_times.append(init_time)
        lsh_search_times.append(search_time)
    lsh_init_times = np.array(lsh_init_times)
    lsh_search_times = np.array(lsh_search_times)

    # Plot initialization time
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(bucket_nums, lsh_init_times, marker='o', label="Initialization Time", color='tab:blue')
    ax1.set_xlabel("Dimension of Projection Set")
    ax1.set_ylabel("Initialization Time (s)", color='tab:blue')
    ax1.set_ylim(min(lsh_init_times) * 0.95, max(lsh_init_times) * 1.05)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot search time
    ax2 = ax1.twinx()
    ax2.plot(bucket_nums, lsh_search_times, marker='o', label="Search Time", color='tab:red')
    ax2.set_ylabel("Search Time (s)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Plot settings
    plt.grid(True, alpha=0.5)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 0.97))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_lsh_knn(
        bucket_nums: range = range(1, 25), 
        repeat_times: int = 100,
        save_path: str = "results/time_comp.png"
):
    """
    Compare the searching speedup of LSH over KNN
    """
    # Compute the time for KNN search
    knn_time = compute_knn_time(repeat_times)
    
    # Compute the time for LSH search
    lsh_times = []
    for bucket_num in tqdm(bucket_nums, desc="Different Dimension of Projection Set"):
        indicators = generate_scatter_indicator(bucket_num)
        _, lsh_time = compute_lsh_time(indicators, (1, repeat_times))
        lsh_times.append(lsh_time)
    lsh_times = np.array(lsh_times)

    # Compute the speedup
    speedup = knn_time / lsh_times

    # Plot the speedup
    fig = plt.figure(figsize=(8, 6))
    plt.plot(bucket_nums, speedup, marker='o')
    plt.xlabel("Dimension of Projection Set")
    plt.ylabel("Speedup")
    plt.grid(alpha=0.5)
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_similarity(
        resnet: bool = False
):
    """
    Plot the similarity between the target image and all dataset images
    """
    # Compute feature vector for the target image
    target_img = Image("target.jpg", resnet)
    target_vec = target_img.feature_vec
    target_vec = target_vec / np.linalg.norm(target_vec)

    # Compute similarity between the target image and all dataset images
    similarities = []
    for i in range(1, 41):
        img = Image(f"dataset/{i}.jpg", resnet)
        img_vec = img.feature_vec
        img_vec = img_vec / np.linalg.norm(img_vec)
        similarities.append(np.dot(target_vec, img_vec))
        
    # Plot the similarity
    fig = plt.figure(figsize=(10, 5))
    plt.bar(range(1, 41), similarities)
    plt.xlabel("Image Index")
    plt.ylabel("Similarity with Target Image")
    plt.xlim(0, 41)
    plt.xticks(range(1, 41), fontsize=10) 
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    save_path = "results/similarity_new.png" if not resnet else "results/similarity_resnet_new.png"
    plt.savefig(save_path, dpi=300)
    plt.show()


def main(args):
    if args[0] == "time_comp":
        plot_lsh_knn(range(1, 25), 100, "results/time_comp_new.png")
    elif args[0] == "time_init_search":
        plot_init_search(range(1, 25), (500, 100), "results/time_init_search_new.png")
    elif args[0] == "similarity":
        plot_similarity(resnet=False)
    elif args[0] == "similarity_resnet":
        plot_similarity(resnet=True)


if __name__ == '__main__':
    main(sys.argv[1:])