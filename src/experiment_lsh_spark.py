#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, MinHashLSH, RegexTokenizer, StopWordsRemover
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, IntegerType, StringType, StructField, StructType

LOGGER = logging.getLogger("spark_lsh")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "training.1600000.processed.noemoticon.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "lsh_spark"

DEFAULT_STOP_WORDS = {
    "rt",
    "amp",
    "https",
    "http",
    "co",
    "im",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinHash LSH experiments with PySpark.")
    parser.add_argument(
        "--data-path",
        type=Path,
    default=DEFAULT_DATA_PATH,
        help="Path to the Sentiment140 CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV/Markdown outputs will be saved.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Sequential row limit before sampling (0 = entire file).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=6000,
        help="Approximate number of tweets to keep per class after sampling (use 0 to keep all).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of data reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling and splits.",
    )
    parser.add_argument(
        "--num-hash-tables",
        type=int,
        nargs="*",
        default=[128, 250],
        help="List of numHashTables values to test.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="*",
        default=[50, 100, 150, 200],
        help="List of k (nearest neighbours) values to test.",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=1 << 15,
        help="Size of the hashing space used by HashingTF.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=1.0,
        help="Maximum Jaccard distance when pairing test/train tweets (1.0 keeps every candidate).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity level for logging.",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_spark(app_name: str = "Sentiment140-MinHashSpark") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.maxResultSize", "4g")
        .getOrCreate()
    )


def load_dataset(spark: SparkSession, args: argparse.Namespace) -> DataFrame:
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")

    schema = StructType(
        [
            StructField("label_raw", IntegerType(), True),
            StructField("tweet_id", StringType(), True),
            StructField("date", StringType(), True),
            StructField("query", StringType(), True),
            StructField("user", StringType(), True),
            StructField("text", StringType(), True),
        ]
    )

    df = (
        spark.read.csv(str(args.data_path), schema=schema, header=False, encoding="ISO-8859-1")
        .select("label_raw", "text")
        .dropna(subset=["text", "label_raw"])
    )

    if args.max_rows and args.max_rows > 0:
        df = df.limit(args.max_rows)

    df = df.withColumn(
        "label",
        F.when(F.col("label_raw") == 0, F.lit(0))
        .when(F.col("label_raw") == 4, F.lit(1))
        .otherwise(F.lit(None))
        .cast(IntegerType()),
    ).dropna(subset=["label"])

    LOGGER.info("Spark DF loaded: %d rows", df.count())
    return df.select("label", "text")


def class_balanced_sample(df: DataFrame, per_class: int, seed: int) -> DataFrame:
    if per_class <= 0:
        return df

    counts = {row["label"]: row["count"] for row in df.groupBy("label").count().collect()}
    fractions: Dict[int, float] = {}
    for label, count in counts.items():
        fraction = min(1.0, per_class / count)
        if fraction <= 0:
            fraction = 1.0
        fractions[label] = fraction
    LOGGER.info("Sampling fractions per label: %s", fractions)
    return df.sampleBy("label", fractions=fractions, seed=seed)


def build_feature_pipeline(num_features: int) -> Pipeline:
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens_raw",
        pattern=r"[^\w#@']+",
        toLowercase=True,
    )
    remover = StopWordsRemover(
        inputCol="tokens_raw",
        outputCol="tokens",
        stopWords=sorted(set(StopWordsRemover.loadDefaultStopWords("english")) | DEFAULT_STOP_WORDS),
    )
    hashing_tf = HashingTF(
        inputCol="tokens",
        outputCol="features",
        numFeatures=num_features,
        binary=True,
    )
    return Pipeline(stages=[tokenizer, remover, hashing_tf])


def evaluate_configuration(
    train_df: DataFrame,
    test_df: DataFrame,
    train_size: int,
    test_size: int,
    num_perm: int,
    k_neighbors: int,
    similarity_threshold: float,
    fallback_label: int,
) -> Dict[str, float]:
    LOGGER.info("Spark LSH | numHashTables=%d | k=%d", num_perm, k_neighbors)
    lsh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=num_perm)
    start_train = time.perf_counter()
    model = lsh.fit(train_df)
    training_time = time.perf_counter() - start_train

    start_eval = time.perf_counter()
    joined = model.approxSimilarityJoin(test_df, train_df, similarity_threshold, distCol="distance")

    pairs = joined.select(
        F.col("datasetA.row_id").alias("test_id"),
        F.col("datasetA.label").alias("test_label"),
        F.col("datasetB.label").alias("train_label"),
        F.col("distance"),
    )

    w = Window.partitionBy("test_id").orderBy("distance")
    topk = pairs.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= k_neighbors)
    topk = topk.persist(StorageLevel.MEMORY_AND_DISK)

    vote_counts = (
        topk.groupBy("test_id", "test_label", "train_label")
        .count()
        .withColumnRenamed("count", "votes")
    )

    neighbor_rows = vote_counts.agg(F.sum("votes").alias("total_votes")).first()["total_votes"]
    neighbor_rows = float(neighbor_rows or 0)

    vote_window = Window.partitionBy("test_id").orderBy(F.desc("votes"), F.asc("train_label"))
    predictions = (
        vote_counts.withColumn("vote_rank", F.row_number().over(vote_window))
        .filter(F.col("vote_rank") == 1)
        .select(
            F.col("test_id"),
            F.col("train_label").alias("prediction"),
        )
    )

    test_labels = test_df.select(
        F.col("row_id").alias("test_id"),
        F.col("label").alias("test_label"),
    )

    preds_full = (
        test_labels.join(predictions, on="test_id", how="left")
        .fillna(fallback_label, subset=["prediction"])
    )

    correct = preds_full.filter(F.col("prediction") == F.col("test_label")).count()
    accuracy = correct / test_size if test_size else 0.0
    avg_candidates = neighbor_rows / test_size if test_size else 0.0
    topk.unpersist()
    eval_time = time.perf_counter() - start_eval

    LOGGER.info(
        "Spark LSH | numHashTables=%d | k=%d | accuracy=%.4f | avg_neighbors=%.1f",
        num_perm,
        k_neighbors,
        accuracy,
        avg_candidates or 0,
    )

    return {
        "num_hash_tables": num_perm,
        "k": k_neighbors,
        "accuracy": accuracy,
        "avg_candidates": float(avg_candidates or 0),
        "training_time_sec": training_time,
        "evaluation_time_sec": eval_time,
        "neighbor_pairs": neighbor_rows,
        "test_size": test_size,
        "train_size": train_size,
    }


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_tables(df: pd.DataFrame, output_dir: Path) -> None:
    csv_path = output_dir / "spark_lsh_results.csv"
    md_path = output_dir / "spark_lsh_results.md"
    df.to_csv(csv_path, index=False)

    md_lines = [
        "| numHashTables | k | accuracy | avg_candidates | train_time_s | eval_time_s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in df.itertuples(index=False):
        md_lines.append(
            f"| {row.num_hash_tables} | {row.k} | {row.accuracy:.4f} | {row.avg_candidates:.1f} | "
            f"{row.training_time_sec:.2f} | {row.evaluation_time_sec:.2f} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    LOGGER.info("Saved Spark results to %s and %s", csv_path, md_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = load_dataset(spark, args)
    df = class_balanced_sample(df, args.samples_per_class, args.random_state)
    df = df.dropDuplicates(["text"])
    df = df.withColumn("row_id", F.monotonically_increasing_id())

    pipeline = build_feature_pipeline(args.num_features)
    features_model = pipeline.fit(df)
    has_tokens = F.udf(lambda v: v is not None and v.numNonzeros() > 0, BooleanType())
    features_df = (
        features_model.transform(df)
        .select("row_id", "label", "features")
        .filter(has_tokens("features"))
    )

    train_df, test_df = features_df.randomSplit(
        [1 - args.test_fraction, args.test_fraction], seed=args.random_state
    )
    train_df = train_df.cache()
    test_df = test_df.cache()
    train_size = train_df.count()
    test_size = test_df.count()
    LOGGER.info("Train size=%d | Test size=%d", train_size, test_size)

    fallback_label = (
        train_df.groupBy("label")
        .count()
        .orderBy(F.desc("count"))
        .first()["label"]
    )

    results: List[Dict[str, float]] = []
    for num_perm in args.num_hash_tables:
        for k_neighbors in args.k_values:
            res = evaluate_configuration(
                train_df,
                test_df,
                train_size=train_size,
                test_size=test_size,
                num_perm=num_perm,
                k_neighbors=k_neighbors,
                similarity_threshold=args.similarity_threshold,
                fallback_label=fallback_label,
            )
            results.append(res)

    spark.stop()

    ensure_output_dir(args.output_dir)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(["num_hash_tables", "k"]).reset_index(drop=True)
    save_tables(df_results, args.output_dir)

    best = df_results.sort_values("accuracy", ascending=False).iloc[0]
    LOGGER.info(
        "Best Spark config: numHashTables=%d | k=%d | accuracy=%.4f",
        best.num_hash_tables,
        best.k,
        best.accuracy,
    )


if __name__ == "__main__":
    main()
