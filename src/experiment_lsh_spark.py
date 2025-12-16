#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import zlib
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, IntegerType, StringType, StructField, StructType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.types import ArrayType
from pyspark.ml.feature import (
    HashingTF, 
    MinHashLSH, 
    RegexTokenizer, 
    StopWordsRemover, 
    NGram, 
    CountVectorizer, 
    VectorAssembler
)
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
        "--test-path",
        type=Path,
        default=None,
        help="Optional additional CSV (e.g., test.csv) to append before splitting.",
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
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="1=Unigram, 2=Bigram, 3=Frequent Patterns, 4=Combined (1+2+3)",
    )

    parser.add_argument(
        "--use-neutral",
        default=False,
        action="store_true",
        help="Include neutral tweets (label=2) in the dataset.",
    )

    parser.add_argument(
        "--dataset-format",
        type=str,
        default="sentiment140",
        choices=["sentiment140", "tweetextraction"],
        help="Dataset schema: 'sentiment140' (6 cols, labels 0/4[/2]) or 'tweetextraction' (textID,text,selected_text,sentiment)",
    )

    return parser.parse_args(argv)

def clean_text_column(df: DataFrame, input_col: str = "text", output_col: str = "text_clean") -> DataFrame:
    c = F.col(input_col)

    # Lowercase (explicite)
    c = F.lower(c)

    # URLs (http(s) + www)
    c = F.regexp_replace(c, r"(?i)\b(?:https?://|www\.)\S+\b", " ")

    # HTML entities fréquentes
    c = F.regexp_replace(c, r"&amp;", " and ")
    c = F.regexp_replace(c, r"&quot;|&lt;|&gt;", " ")
    c = F.regexp_replace(c, r"&#39;", "'")

    # Retweets / mentions
    c = F.regexp_replace(c, r"(?i)^\s*rt\s+", " ")
    c = F.regexp_replace(c, r"@[A-Za-z0-9_]+", " ")

    # Hashtags: garder le mot sans '#'
    c = F.regexp_replace(c, r"#(\w+)", r"\1")

    # Remplacer la plupart des caractères non utiles par espace
    # (on garde lettres/chiffres/_/apostrophe et espaces)
    c = F.regexp_replace(c, r"[^a-z0-9_'\s]", " ")

    # Espaces multiples + trim
    c = F.regexp_replace(c, r"\s+", " ")
    c = F.trim(c)

    return df.withColumn(output_col, c)

def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_spark(app_name: str = "Sentiment140-MinHashSpark") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "800")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )


def load_dataset(spark: SparkSession, args: argparse.Namespace, use_neutral: bool) -> DataFrame:
    def load_sentiment140(path: Path) -> DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

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

        df_local = (
            spark.read.csv(str(path), schema=schema, header=False, encoding="ISO-8859-1")
            .select("label_raw", "text")
            .dropna(subset=["text", "label_raw"])
        )
        if args.max_rows and args.max_rows > 0:
            df_local = df_local.limit(args.max_rows)

        if use_neutral:
            df_local = df_local.withColumn(
                "label",
                F.when(F.col("label_raw") == 0, F.lit(0))
                .when(F.col("label_raw") == 2, F.lit(1))
                .when(F.col("label_raw") == 4, F.lit(2))
                .otherwise(F.lit(None))
                .cast(IntegerType()),
            ).dropna(subset=["label"])
        else:
            df_local = df_local.withColumn(
                "label",
                F.when(F.col("label_raw") == 0, F.lit(0))
                .when(F.col("label_raw") == 4, F.lit(1))
                .otherwise(F.lit(None))
                .cast(IntegerType()),
            ).dropna(subset=["label"])

        return df_local.select("label", "text")

    def load_tweetextraction(path: Path) -> DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        df_local = (
            spark.read.csv(str(path), header=True, inferSchema=True)
            .select(F.col("sentiment").alias("sentiment_str"), F.col("text"))
            .dropna(subset=["text", "sentiment_str"])
        )
        if args.max_rows and args.max_rows > 0:
            df_local = df_local.limit(args.max_rows)

        if use_neutral:
            df_local = df_local.withColumn(
                "label",
                F.when(F.lower(F.col("sentiment_str")) == "negative", F.lit(0))
                .when(F.lower(F.col("sentiment_str")) == "neutral", F.lit(1))
                .when(F.lower(F.col("sentiment_str")) == "positive", F.lit(2))
                .otherwise(F.lit(None))
                .cast(IntegerType()),
            ).dropna(subset=["label"])
        else:
            df_local = df_local.withColumn(
                "label",
                F.when(F.lower(F.col("sentiment_str")) == "negative", F.lit(0))
                .when(F.lower(F.col("sentiment_str")) == "positive", F.lit(1))
                .otherwise(F.lit(None))
                .cast(IntegerType()),
            ).dropna(subset=["label"])

        return df_local.select("label", "text")

    loader = load_sentiment140 if args.dataset_format == "sentiment140" else load_tweetextraction

    df = loader(args.data_path)
    if args.test_path is not None:
        df_extra = loader(args.test_path)
        df = df.unionByName(df_extra)

    LOGGER.info("Spark DF loaded: %d rows", df.count())
    return df


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

class HFWPatternizer(
    Estimator,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Estimator:
    - fit(): calcule les High-Frequency Words (HFW) = topN tokens les plus fréquents
    - retourne un Model qui transforme tokens -> patterns (HFW gardés, CW remplacés par '*')
    """
    def __init__(
        self,
        inputCol: str = "tokens",
        outputCol: str = "patterns",
        topN: int = 300,
        minN: int = 3,
        maxN: int = 6,
        minHFWInWindow: int = 2,
        maxPatternsPerDoc: int = 80,
    ):
        super().__init__()
        self._setDefault(
            inputCol=inputCol,
            outputCol=outputCol,
            topN=topN,
            minN=minN,
            maxN=maxN,
            minHFWInWindow=minHFWInWindow,
            maxPatternsPerDoc=maxPatternsPerDoc,
        )
        self._set(inputCol=inputCol, outputCol=outputCol)
        self.topN = topN
        self.minN = minN
        self.maxN = maxN
        self.minHFWInWindow = minHFWInWindow
        self.maxPatternsPerDoc = maxPatternsPerDoc

    def _fit(self, dataset: DataFrame):
        topN = int(self.topN)
        inputCol = self.getInputCol()

        # fréquence des tokens
        freq = (
            dataset.select(F.explode(F.col(inputCol)).alias("tok"))
            .groupBy("tok")
            .count()
            .orderBy(F.desc("count"))
            .limit(topN)
        )

        hfw = [r["tok"] for r in freq.collect()]
        return HFWPatternizerModel(
            hfw_words=hfw,
            inputCol=self.getInputCol(),
            outputCol=self.getOutputCol(),
            minN=int(self.minN),
            maxN=int(self.maxN),
            minHFWInWindow=int(self.minHFWInWindow),
            maxPatternsPerDoc=int(self.maxPatternsPerDoc),
        )


class HFWPatternizerModel(
    Model,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    def __init__(
    self,
    inputCol: str = "tokens",
    outputCol: str = "patterns",
    topN: int = 300,
    minN: int = 3,
    maxN: int = 6,
    minHFWInWindow: int = 2,
    maxPatternsPerDoc: int = 80,
):
        super().__init__()
        self._set(inputCol=inputCol, outputCol=outputCol)
        self.topN = topN
        self.minN = minN
        self.maxN = maxN
        self.minHFWInWindow = minHFWInWindow
        self.maxPatternsPerDoc = maxPatternsPerDoc

    def _transform(self, dataset: DataFrame) -> DataFrame:
        hfw_set = set(self.hfw_words)
        minN = self.minN
        maxN = self.maxN
        minHFW = self.minHFWInWindow
        maxP = self.maxPatternsPerDoc

        def make_patterns(tokens):
            if not tokens:
                return []
            # stream HFW / *
            stream = [t if t in hfw_set else "*" for t in tokens]
            out = []
            seen = set()

            L = len(stream)
            for n in range(minN, maxN + 1):
                if n > L:
                    break
                for i in range(L - n + 1):
                    win = stream[i : i + n]
                    hfw_count = sum(1 for w in win if w != "*")
                    if hfw_count < minHFW:
                        continue
                    p = "_".join(win)
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
                        if len(out) >= maxP:
                            return out
            return out

        udf_patterns = F.udf(make_patterns, ArrayType(StringType()))
        return dataset.withColumn(self.getOutputCol(), udf_patterns(F.col(self.getInputCol())))
    
def bloom_filter_func(tokens, vector_size, num_hashes):
    """
    Transforme une liste de tokens en un vecteur binaire représentant un Filtre de Bloom.
    
    Principe :
    Contrairement à HashingTF qui hache un mot vers 1 seul index, 
    le filtre de Bloom hache un mot vers 'num_hashes' index différents.
    Cela augmente la robustesse contre les collisions.
    """
    if not tokens:
        return Vectors.sparse(vector_size, [], [])
    
    indices = set()
    
    for token in tokens:
        # Conversion en bytes pour le hachage
        b = token.encode('utf-8')
        
        # Nous utilisons deux fonctions de hachage de base : CRC32 et Adler32
        # Ce sont des fonctions rapides et déterministes.
        h1 = zlib.crc32(b)
        h2 = zlib.adler32(b)
        
        # Astuce de Kirsch-Mitzenmacher :
        # Au lieu de calculer k vrais hachages (lents), on simule k hachages
        # avec la formule : g_i(x) = h1(x) + i * h2(x)
        for i in range(num_hashes):
            idx = (h1 + i * h2) % vector_size
            indices.add(idx)
            
    # On retourne un SparseVector compatible avec PySpark ML
    # Les valeurs sont toutes à 1.0 (présence binaire)
    sorted_indices = sorted(list(indices))
    return Vectors.sparse(vector_size, sorted_indices, [1.0] * len(sorted_indices))

def build_feature_pipeline(num_features: int, scenario: int) -> Pipeline:
    tokenizer = RegexTokenizer(
        inputCol="text_clean",
        outputCol="tokens_raw",
        pattern=r"[^\w#@']+",
        toLowercase=True,
    )
    
    remover = StopWordsRemover(
        inputCol="tokens_raw",
        outputCol="tokens",
        stopWords=sorted(set(StopWordsRemover.loadDefaultStopWords("english")) | DEFAULT_STOP_WORDS),
    )

    stages = [tokenizer, remover]

    if scenario == 1:
        hashing_tf = HashingTF(
            inputCol="tokens",
            outputCol="features",
            numFeatures=num_features,
            binary=True,
        )
        stages.append(hashing_tf)

    elif scenario == 2:
        ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
        hashing_tf = HashingTF(
            inputCol="bigrams",
            outputCol="features",
            numFeatures=num_features,
            binary=True,
        )
        stages.extend([ngram, hashing_tf])

    elif scenario == 3:
        # Patterns au sens "HFW + slots * + sous-séquences"
        patternizer = HFWPatternizer(
            inputCol="tokens",
            outputCol="patterns",
            topN=300,        # ajuste si besoin
            minN=3,
            maxN=6,
            minHFWInWindow=2,
            maxPatternsPerDoc=80,
        )
        cv = CountVectorizer(
            inputCol="patterns",
            outputCol="features",
            vocabSize=20000,
            binary=True
        )
        stages.extend([patternizer, cv])

    elif scenario == 4:
        # Unigram
        htf_uni = HashingTF(inputCol="tokens", outputCol="vec_uni", numFeatures=num_features, binary=True)

        # Bigram
        ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
        htf_bi = HashingTF(inputCol="bigrams", outputCol="vec_bi", numFeatures=num_features, binary=True)

        # Patterns
        patternizer = HFWPatternizer(
            inputCol="tokens",
            outputCol="patterns",
            topN=300,
            minN=3,
            maxN=6,
            minHFWInWindow=2,
            maxPatternsPerDoc=80,
        )
        cv_pat = CountVectorizer(
            inputCol="patterns",
            outputCol="vec_pat",
            vocabSize=20000,
            binary=True
        )

        assembler = VectorAssembler(
            inputCols=["vec_uni", "vec_bi", "vec_pat"],
            outputCol="features"
        )

        stages.extend([htf_uni, ngram, htf_bi, patternizer, cv_pat, assembler])

    return Pipeline(stages=stages)


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
        F.col("datasetB.row_id").alias("train_id"),
        F.col("datasetB.label").alias("train_label"),
        F.col("distance"),
    ).dropDuplicates(["test_id", "train_id"])

    pairs = pairs.repartition("test_id")

    w = Window.partitionBy("test_id").orderBy("distance")
    topk = (pairs
            .withColumn("rank", F.row_number().over(w))
            .filter(F.col("rank") <= k_neighbors)
            .select("test_id", "test_label", "train_label")
            .persist(StorageLevel.DISK_ONLY)
    )
    topk = topk.persist(StorageLevel.MEMORY_AND_DISK)

    vote_counts = (
        topk.groupBy("test_id", "test_label", "train_label")
        .count()
        .withColumnRenamed("count", "votes")
    )

    neighbor_rows = float(topk.count())

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

    df = load_dataset(spark, args, args.use_neutral)
    df = clean_text_column(df, input_col="text", output_col="text_clean")
    df = class_balanced_sample(df, args.samples_per_class, args.random_state)
    df = df.dropDuplicates(["text_clean"])
    df = df.withColumn("row_id", F.monotonically_increasing_id())

    pipeline = build_feature_pipeline(args.num_features, args.scenario)
    features_model = pipeline.fit(df)
    has_tokens = F.udf(lambda v: v is not None and v.numNonzeros() > 0, BooleanType())
    features_df = (
        features_model.transform(df)
        .select("row_id", "label", "features")
        # Filtre les vecteurs vides pour éviter l'erreur MinHash "Must have at least 1 non zero entry"
        .filter(has_tokens("features"))
    )

    tmp = features_model.transform(df).select("text_clean", "tokens", "patterns").limit(5)
    tmp.show(truncate=120)

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
