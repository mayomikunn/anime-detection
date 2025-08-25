# **Anime Detection Project**
## Overview

This project is a content-based image retrieval (CBIR) system designed to identify the anime series from which a given clip or frame originates. By combining techniques from computer vision and image processing, the system compares input images against a large dataset of anime images to find the closest matches.

The project was built as part of my final-year university work, but it also reflects my personal interest in applying machine learning principles, computer vision, and software engineering to solve real-world problems.

## Tech Stack

Programming Language: Python

Libraries & Tools:

OpenCV (image processing, edge detection)

scikit-learn (K-means clustering, similarity metrics)

NumPy, SciPy (matrix operations, distance computations)

Matplotlib (visualisation of results)

## Methodology

Database Preprocessing:

Extract dominant colours, build histograms, and compute edge maps for all images.

Cluster images to reduce retrieval time.

Query Matching:

Preprocess query image using same pipeline.

Identify the closest cluster based on dominant colour similarity.

Compare against preprocessed features using multiple similarity metrics.

Output:

Return the most similar anime images alongside a similarity percentage.

(Future work: integrate with AniList API to retrieve anime descriptions.)
