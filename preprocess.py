"""
Data preprocessing and exploration utilities.

This file is a required deliverable for the Dacon competition.
It provides utilities for analyzing the dataset, computing statistics,
and preparing data for training.
"""

import os
import argparse
import logging
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.io import read_json_annotation

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess and analyze dataset"
    )

    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to JSON annotation file",
    )
    parser.add_argument(
        "--jpg_dir",
        type=str,
        required=True,
        help="Path to images directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )

    return parser.parse_args()


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def analyze_dataset(json_path: str, jpg_dir: str) -> Dict:
    """
    Analyze dataset and compute statistics.

    Args:
        json_path: Path to JSON annotation file
        jpg_dir: Path to images directory

    Returns:
        Dictionary containing dataset statistics
    """
    logger.info("Loading annotations...")
    annotations = read_json_annotation(json_path)

    stats = {
        "total_images": len(annotations),
        "total_annotations": 0,
        "visual_annotations": 0,
        "text_annotations": 0,
        "queries": [],
        "bbox_sizes": [],
        "image_sizes": [],
        "category_distribution": Counter(),
    }

    logger.info("Analyzing annotations...")
    for item in tqdm(annotations, desc="Processing"):
        # Get annotations from learning_data_info
        learning_data_info = item.get("learning_data_info", {})
        anns = learning_data_info.get("annotation", [])

        stats["total_annotations"] += len(anns)

        for ann in anns:
            class_id = ann.get("class_id", "")
            class_name = ann.get("class_name", "")

            # Determine if visual or text annotation based on class_id
            is_visual = class_id.startswith("V") or any(
                k in class_name for k in ["표", "차트", "그래프", "그림"]
            )

            if is_visual:
                stats["visual_annotations"] += 1

                # Collect query information from visual_instruction
                query = ann.get("visual_instruction", "")
                if query:
                    stats["queries"].append(query)

                # Collect bbox information
                bbox = ann.get("bounding_box")
                if bbox:
                    x, y, w, h = bbox
                    stats["bbox_sizes"].append((w, h))
            else:
                stats["text_annotations"] += 1

            # Track class distribution
            stats["category_distribution"][class_id] += 1

        # Collect image size information from source_data_info
        source_data_info = item.get("source_data_info", {})
        source_data_name_jpg = source_data_info.get("source_data_name_jpg", "")

        if source_data_name_jpg and os.path.exists(
            os.path.join(jpg_dir, source_data_name_jpg)
        ):
            try:
                img_path = os.path.join(jpg_dir, source_data_name_jpg)
                img = Image.open(img_path)
                stats["image_sizes"].append(img.size)
            except Exception as e:
                logger.warning(f"Failed to load image {source_data_name_jpg}: {e}")

    return stats


def compute_query_statistics(queries: List[str]) -> Dict:
    """
    Compute statistics for text queries.

    Args:
        queries: List of query strings

    Returns:
        Dictionary with query statistics
    """
    query_lengths = [len(q.split()) for q in queries]
    char_lengths = [len(q) for q in queries]

    # Count unique queries
    unique_queries = len(set(queries))

    # Count token frequencies
    all_tokens = []
    for q in queries:
        all_tokens.extend(q.split())

    token_freq = Counter(all_tokens)
    vocab_size = len(token_freq)

    return {
        "total_queries": len(queries),
        "unique_queries": unique_queries,
        "avg_query_length_words": np.mean(query_lengths),
        "max_query_length_words": np.max(query_lengths),
        "min_query_length_words": np.min(query_lengths),
        "avg_query_length_chars": np.mean(char_lengths),
        "vocabulary_size": vocab_size,
        "most_common_tokens": token_freq.most_common(20),
    }


def compute_bbox_statistics(bbox_sizes: List[Tuple[float, float]]) -> Dict:
    """
    Compute statistics for bounding boxes.

    Args:
        bbox_sizes: List of (width, height) tuples

    Returns:
        Dictionary with bbox statistics
    """
    widths = [w for w, h in bbox_sizes]
    heights = [h for w, h in bbox_sizes]
    areas = [w * h for w, h in bbox_sizes]
    aspect_ratios = [w / h if h > 0 else 0 for w, h in bbox_sizes]

    return {
        "total_bboxes": len(bbox_sizes),
        "avg_width": np.mean(widths),
        "avg_height": np.mean(heights),
        "avg_area": np.mean(areas),
        "avg_aspect_ratio": np.mean(aspect_ratios),
        "min_width": np.min(widths),
        "max_width": np.max(widths),
        "min_height": np.min(heights),
        "max_height": np.max(heights),
    }


def compute_image_statistics(image_sizes: List[Tuple[int, int]]) -> Dict:
    """
    Compute statistics for image dimensions.

    Args:
        image_sizes: List of (width, height) tuples

    Returns:
        Dictionary with image statistics
    """
    widths = [w for w, h in image_sizes]
    heights = [h for w, h in image_sizes]

    return {
        "total_images": len(image_sizes),
        "avg_width": np.mean(widths),
        "avg_height": np.mean(heights),
        "min_width": np.min(widths),
        "max_width": np.max(widths),
        "min_height": np.min(heights),
        "max_height": np.max(heights),
    }


def print_statistics(stats: Dict) -> None:
    """
    Print dataset statistics in a readable format.

    Args:
        stats: Dictionary containing dataset statistics
    """
    logger.info("\n" + "="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)

    logger.info(f"\nGeneral Statistics:")
    logger.info(f"  Total Images: {stats['total_images']}")
    logger.info(f"  Total Annotations: {stats['total_annotations']}")
    logger.info(f"  Visual Annotations (category_id=1): {stats['visual_annotations']}")
    logger.info(f"  Text Annotations (category_id=2): {stats['text_annotations']}")

    if stats['queries']:
        query_stats = compute_query_statistics(stats['queries'])
        logger.info(f"\nQuery Statistics:")
        logger.info(f"  Total Queries: {query_stats['total_queries']}")
        logger.info(f"  Unique Queries: {query_stats['unique_queries']}")
        logger.info(f"  Vocabulary Size: {query_stats['vocabulary_size']}")
        logger.info(f"  Avg Query Length (words): {query_stats['avg_query_length_words']:.2f}")
        logger.info(f"  Max Query Length (words): {query_stats['max_query_length_words']}")
        logger.info(f"  Min Query Length (words): {query_stats['min_query_length_words']}")
        logger.info(f"  Avg Query Length (chars): {query_stats['avg_query_length_chars']:.2f}")

        logger.info(f"\n  Most Common Tokens:")
        for token, count in query_stats['most_common_tokens'][:10]:
            logger.info(f"    '{token}': {count}")

    if stats['bbox_sizes']:
        bbox_stats = compute_bbox_statistics(stats['bbox_sizes'])
        logger.info(f"\nBounding Box Statistics:")
        logger.info(f"  Total Bboxes: {bbox_stats['total_bboxes']}")
        logger.info(f"  Avg Width: {bbox_stats['avg_width']:.2f} px")
        logger.info(f"  Avg Height: {bbox_stats['avg_height']:.2f} px")
        logger.info(f"  Avg Area: {bbox_stats['avg_area']:.2f} px²")
        logger.info(f"  Avg Aspect Ratio: {bbox_stats['avg_aspect_ratio']:.2f}")
        logger.info(f"  Width Range: [{bbox_stats['min_width']:.0f}, {bbox_stats['max_width']:.0f}] px")
        logger.info(f"  Height Range: [{bbox_stats['min_height']:.0f}, {bbox_stats['max_height']:.0f}] px")

    if stats['image_sizes']:
        img_stats = compute_image_statistics(stats['image_sizes'])
        logger.info(f"\nImage Statistics:")
        logger.info(f"  Images Analyzed: {img_stats['total_images']}")
        logger.info(f"  Avg Width: {img_stats['avg_width']:.2f} px")
        logger.info(f"  Avg Height: {img_stats['avg_height']:.2f} px")
        logger.info(f"  Width Range: [{img_stats['min_width']:.0f}, {img_stats['max_width']:.0f}] px")
        logger.info(f"  Height Range: [{img_stats['min_height']:.0f}, {img_stats['max_height']:.0f}] px")

    logger.info("\n" + "="*60)


def save_statistics(stats: Dict, output_dir: str) -> None:
    """
    Save statistics to files.

    Args:
        stats: Dictionary containing dataset statistics
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save query statistics
    if stats['queries']:
        query_stats = compute_query_statistics(stats['queries'])

        # Save to text file
        with open(os.path.join(output_dir, "query_stats.txt"), "w", encoding="utf-8") as f:
            f.write("Query Statistics\n")
            f.write("="*60 + "\n\n")
            for key, value in query_stats.items():
                if key != "most_common_tokens":
                    f.write(f"{key}: {value}\n")

        # Save most common tokens to CSV
        token_df = pd.DataFrame(
            query_stats['most_common_tokens'],
            columns=["token", "count"]
        )
        token_df.to_csv(
            os.path.join(output_dir, "token_frequencies.csv"),
            index=False
        )

    # Save bbox statistics
    if stats['bbox_sizes']:
        bbox_stats = compute_bbox_statistics(stats['bbox_sizes'])
        with open(os.path.join(output_dir, "bbox_stats.txt"), "w", encoding="utf-8") as f:
            f.write("Bounding Box Statistics\n")
            f.write("="*60 + "\n\n")
            for key, value in bbox_stats.items():
                f.write(f"{key}: {value}\n")

        # Save bbox sizes to CSV
        bbox_df = pd.DataFrame(stats['bbox_sizes'], columns=["width", "height"])
        bbox_df.to_csv(
            os.path.join(output_dir, "bbox_sizes.csv"),
            index=False
        )

    # Save image statistics
    if stats['image_sizes']:
        img_stats = compute_image_statistics(stats['image_sizes'])
        with open(os.path.join(output_dir, "image_stats.txt"), "w", encoding="utf-8") as f:
            f.write("Image Statistics\n")
            f.write("="*60 + "\n\n")
            for key, value in img_stats.items():
                f.write(f"{key}: {value}\n")

    logger.info(f"Statistics saved to {output_dir}/")


def main():
    """Main preprocessing function."""
    args = parse_args()
    setup_logging()

    logger.info("="*60)
    logger.info("Dataset Preprocessing and Analysis")
    logger.info("="*60)
    logger.info(f"JSON: {args.json_path}")
    logger.info(f"Images: {args.jpg_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*60)

    # Analyze dataset
    stats = analyze_dataset(args.json_path, args.jpg_dir)

    # Print statistics
    print_statistics(stats)

    # Save statistics to files
    save_statistics(stats, args.output_dir)

    logger.info("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
