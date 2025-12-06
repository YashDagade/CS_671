from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # TODO: Implement init_centroids
    # *** START YOUR CODE ***
    # Get image dimensions
    H, W, C = image.shape

    # Reshape image to (H*W, C) to get all pixels as rows
    pixels = image.reshape(-1, C)

    # Randomly select num_clusters pixels as initial centroids
    random_indices = np.random.choice(pixels.shape[0], num_clusters, replace=False)
    centroids_init = pixels[random_indices].astype(np.float64)
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # TODO: Implement update_centroids
    # *** START YOUR CODE ***
    # Get image dimensions and reshape to (H*W, C)
    H, W, C = image.shape
    pixels = image.reshape(-1, C).astype(np.float64)
    num_clusters = centroids.shape[0]

    new_centroids = centroids.copy()

    # Iterate for max_iter iterations
    for iteration in range(max_iter):
        # Initialize array to store cluster assignments
        cluster_assignments = np.zeros(pixels.shape[0], dtype=int)

        # For each pixel, find the closest centroid using Euclidean distance
        for i in range(pixels.shape[0]):
            # Initialize `dist` vector to keep track of distance to every centroid
            dist = np.zeros(num_clusters)

            # Loop over all centroids and store distances in `dist`
            for j in range(num_clusters):
                dist[j] = np.sqrt(np.sum((pixels[i] - new_centroids[j]) ** 2))

            # Find closest centroid and update `new_centroids`
            cluster_assignments[i] = np.argmin(dist)

        # Update `new_centroids` by computing mean of pixels in each cluster
        old_centroids = new_centroids.copy()
        for j in range(num_clusters):
            cluster_pixels = pixels[cluster_assignments == j]
            if len(cluster_pixels) > 0:
                new_centroids[j] = cluster_pixels.mean(axis=0)

        # Print progress
        if (iteration + 1) % print_every == 0:
            print(f'Iteration {iteration + 1}/{max_iter} complete')

        # Check for convergence
        if np.allclose(old_centroids, new_centroids):
            print(f'Converged at iteration {iteration + 1}')
            break
    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # TODO: Implement update_image
    # *** START YOUR CODE ***
    # Get image dimensions
    H, W, C = image.shape
    num_clusters = centroids.shape[0]

    # Create output image
    updated_image = np.zeros_like(image, dtype=np.float64)

    # For each pixel, find the closest centroid and replace with centroid value
    for i in range(H):
        for j in range(W):
            pixel = image[i, j].astype(np.float64)

            # Initialize `dist` vector to keep track of distance to every centroid
            dist = np.zeros(num_clusters)

            # Loop over all centroids and store distances in `dist`
            for k in range(num_clusters):
                dist[k] = np.sqrt(np.sum((pixel - centroids[k]) ** 2))

            # Find closest centroid and update pixel value in `image`
            closest_centroid = np.argmin(dist)
            updated_image[i, j] = centroids[closest_centroid]

    # *** END YOUR CODE ***

    return updated_image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # TODO: Load small image
    # *** START YOUR CODE ***
    image_small = mpimg.imread(image_path_small)
    print(f'Loaded small image: {image_small.shape}')
    # *** END YOUR CODE ***

    # TODO: Initialize centroids
    # *** START YOUR CODE ***
    centroids = init_centroids(num_clusters, image_small)
    print(f'Initialized {num_clusters} centroids')
    # *** END YOUR CODE ***

    # TODO: Update centroids
    # *** START YOUR CODE ***
    centroids = update_centroids(centroids, image_small, max_iter, print_every)
    print('Centroids updated')
    # *** END YOUR CODE ***

    # TODO: Load large image
    # *** START YOUR CODE ***
    image_large = mpimg.imread(image_path_large)
    print(f'Loaded large image: {image_large.shape}')
    # *** END YOUR CODE ***

    # TODO: Update large image with centroids calculated on small image
    # *** START YOUR CODE ***
    image_compressed = update_image(image_large, centroids)
    print('Large image compressed')
    # *** END YOUR CODE ***

    # TODO: Visualize and save compressed image
    # *** START YOUR CODE ***
    # Display and save original small image
    figure_idx += 1
    plt.figure(figure_idx)
    plt.imshow(image_small)
    plt.title('Original Small Image')
    plt.axis('off')
    plt.savefig('peppers_small_original.png', bbox_inches='tight', dpi=150)

    # Display and save original large image
    figure_idx += 1
    plt.figure(figure_idx)
    plt.imshow(image_large)
    plt.title('Original Large Image')
    plt.axis('off')
    plt.savefig('peppers_large_original.png', bbox_inches='tight', dpi=150)

    # Display and save compressed image
    figure_idx += 1
    plt.figure(figure_idx)
    plt.imshow(image_compressed / 255.0)
    plt.title(f'Compressed Image (k={num_clusters})')
    plt.axis('off')
    plt.savefig('peppers_large_compressed.png', bbox_inches='tight', dpi=150)

    print(f'\nImages saved successfully')
    # *** END YOUR CODE ***

    # Calculate and display compression factor
    print("\n" + "=" * 60)
    print("COMPRESSION FACTOR CALCULATION")
    print("=" * 60)

    # Original image representation
    H, W, C = image_large.shape
    num_pixels = H * W
    bytes_per_pixel = 3  # RGB
    original_size_bytes = num_pixels * bytes_per_pixel

    print("\n### Original Image ###")
    print(f"Image dimensions: {H} × {W} pixels")
    print(f"Total pixels: {num_pixels:,}")
    print(f"Bytes per pixel: {bytes_per_pixel} (RGB channels)")
    print(f"Original size: {original_size_bytes:,} bytes")

    # Compressed representation
    # We have k centroids (colors)
    # Each centroid has 3 channels (R, G, B), each 8-bit
    # We need to store: (1) the centroids and (2) the cluster assignment for each pixel

    import math
    centroid_storage = num_clusters * 3  # k centroids, 3 bytes each

    # For cluster assignments, we need log2(k) bits per pixel
    bits_per_assignment = math.ceil(math.log2(num_clusters))
    bytes_per_assignment = bits_per_assignment / 8
    assignment_storage = num_pixels * bytes_per_assignment

    compressed_size_bytes = centroid_storage + assignment_storage

    print("\n### Compressed Image ###")
    print(f"Number of clusters (k): {num_clusters}")
    print(f"Centroid storage: {num_clusters} centroids × 3 bytes = {centroid_storage} bytes")
    print(f"Bits per cluster assignment: {bits_per_assignment} bits (log2({num_clusters}))")
    print(f"Assignment storage: {num_pixels:,} pixels × {bytes_per_assignment} bytes = {assignment_storage:,.0f} bytes")
    print(f"Total compressed size: {compressed_size_bytes:,.0f} bytes")

    # Calculate compression factor
    compression_factor = original_size_bytes / compressed_size_bytes

    print("\n### Compression Factor ###")
    print(f"Compression factor: {original_size_bytes:,} / {compressed_size_bytes:,.0f}")
    print(f"                  = {compression_factor:.2f}")
    print(f"\nThis means the compressed image uses {100/compression_factor:.2f}% of the original size")
    print(f"Space savings: {100 - 100/compression_factor:.2f}%")
    print("=" * 60)

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
