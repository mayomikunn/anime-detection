# **Anime Detection Project**
## Overview

This project is a content-based image retrieval (CBIR) system designed to identify the anime series from which a given clip or frame originates. By combining techniques from computer vision and image processing, the system compares input images against a large dataset of anime images to find the closest matches.

The project was built as part of my final-year university work, but it also reflects my personal interest in applying machine learning principles, computer vision, and software engineering to solve real-world problems.

## Features

Preprocessing and Clustering

Database images are preprocessed once and clustered using K-means based on dominant colours to reduce search space.

Feature Extraction

Colour Features: Histograms with similarity measured via Euclidean Distance.

Keypoint Features: Extracted using feature detection algorithms (SIFT), then compared via descriptor matching.

Query Processing

User-uploaded images are preprocessed in the same way as database images.

Performance Considerations

Precomputation of features improves runtime performance during queries.

## Tech Stack

Programming Language: Python

Libraries & Tools:

OpenCV (image processing, edge detection)

scikit-learn (K-means clustering, similarity metrics)

NumPy, SciPy (matrix operations, distance computations)

Matplotlib (visualisation of results)

## Methodology

Database Preprocessing:

Extract dominant colours and keypoints for all images.

Query Matching:

Preprocess query image using the same pipeline.

Compare keypoint descriptors and colour histograms against preprocessed features.

Output:

Return the most similar anime images alongside a similarity percentage.

Integrate with AniList API to retrieve anime titles and posters.

## Results & Insights

Combining colour histograms and keypoint-based similarity yielded higher accuracy than colour-only

Preprocessing reduced retrieval times significantly.

Keypoint similarity improved robustness when dealing with images that had different scales, orientations, or partial occlusions.
