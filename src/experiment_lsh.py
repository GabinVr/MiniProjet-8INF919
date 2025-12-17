#!/usr/bin/env python3
"""LSH–MinHash experimentation harness for the mini-project (section 2.1).

The script implements the deliverables described in ``tache.txt``:

* Build an LSH–MinHash pipeline on top of the Sentiment140 dataset
* Evaluate multiple configurations of ``numHashTables`` and ``k``
* Measure training/query time, accuracy and a simple scalability proxy
* Export the measurements as CSV/Markdown tables and accuracy/time plots

Run ``python experriment_lsh.py --help`` for available options. The default
configuration samples a manageable subset of tweets so that the experiments can
run locally without a Spark cluster. Use ``--max-rows`` and ``--sample-size`` to
scale up once the pipeline is validated, or ``--demo`` to run on a tiny toy
dataset for sanity checks.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datasketch import MinHash, MinHashLSHForest
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("experriment_lsh")

TOKEN_REGEX = re.compile(r"[a-zA-ZÀ-ÿ0-9#@']+")
STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
	"rt",
	"amp",
	"https",
	"http",
	"co",
	"im",
}


def _dedupe_preserve_order(items: Iterable[str]) -> Tuple[str, ...]:
	seen: OrderedDict[str, None] = OrderedDict()
	for item in items:
		if not item:
			continue
		seen.setdefault(item, None)
	return tuple(seen.keys())


def _generate_ngrams(tokens: Sequence[str], length: int) -> List[str]:
	length = max(1, int(length))
	if length <= 1 or not tokens or len(tokens) < length:
		return []
	return ["_".join(tokens[i : i + length]) for i in range(len(tokens) - length + 1)]


def build_pattern_vocabulary(
	token_series: Iterable[Sequence[str]],
	length: int,
	max_size: int,
) -> Tuple[str, ...]:
	"""Return the ``max_size`` most frequent patterns of a given length."""

	max_size = max(1, int(max_size))
	length = max(2, int(length))
	counter: Counter[str] = Counter()
	for tokens in token_series:
		counter.update(_generate_ngrams(tokens, length))
	return tuple(key for key, _ in counter.most_common(max_size))


def select_feature_tokens(
	tokens: Sequence[str],
	scenario: str,
	ngram_length: int,
	pattern_length: int,
	pattern_vocab: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
	"""Generate the feature token list for the requested scenario."""

	use_unigrams = scenario in {"unigram", "all"}
	use_bigrams = scenario in {"bigram", "all"}
	use_patterns = scenario in {"patterns", "all"}
	features: List[str] = []

	if use_unigrams:
		features.extend(tokens)
	if use_bigrams:
		features.extend(_generate_ngrams(tokens, max(2, ngram_length)))
	if use_patterns:
		patterns = _generate_ngrams(tokens, max(2, pattern_length))
		if pattern_vocab:
			allowed = set(pattern_vocab)
			patterns = [p for p in patterns if p in allowed]
		features.extend(patterns)

	return _dedupe_preserve_order(features)


@dataclass(frozen=True)
class ExperimentConfig:
	"""Configuration tuple for a single experiment."""

	num_perm: int
	k_neighbors: int


@dataclass
class ExperimentResult:
	"""Aggregated metrics captured for an experiment."""

	num_perm: int
	k_neighbors: int
	accuracy: float
	avg_query_time_ms: float
	avg_candidates: float
	training_time_sec: float
	throughput_samples_per_sec: float
	train_size: int
	test_size: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	"""Parse CLI arguments."""

	parser = argparse.ArgumentParser(
		description="Run MinHash LSH experiments on the Sentiment140 dataset."
	)
	parser.add_argument(
		"--data-path",
		type=Path,
		default=Path("data/training.1600000.processed.noemoticon.csv"),
		help="Path to the Sentiment140 CSV file.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("../reports/lsh"),
		help="Directory where CSV tables and PNG plots will be stored.",
	)
	parser.add_argument(
		"--max-rows",
		type=int,
		default=0,
		help="Maximum number of rows to read sequentially from disk (set 0 to stream the entire file).",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=40_000,
		help="Random sample size taken after loading the CSV (<= max rows).",
	)
	parser.add_argument(
		"--test-fraction",
		type=float,
		default=0.2,
		help="Fraction of the sample reserved for evaluation (0-1).",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed used for sampling and train/test split.",
	)
	parser.add_argument(
		"--num-hash-tables",
		type=int,
		nargs="*",
		default=[128, 250],
		help="List of numHashTables (MinHash permutations) to evaluate.",
	)
	parser.add_argument(
		"--k-values",
		type=int,
		nargs="*",
		default=[50, 100, 150, 200],
		help="List of k (nearest neighbours) values to evaluate.",
	)
	parser.add_argument(
		"--feature-scenario",
		type=str,
		choices=["unigram", "bigram", "patterns", "all"],
		default="unigram",
		help="Feature configuration: unigram baseline, bigrams only, frequent patterns or the combined setup.",
	)
	parser.add_argument(
		"--bigram-length",
		type=int,
		default=2,
		help="N-gram length used when generating scenario-2 features (minimum 2).",
	)
	parser.add_argument(
		"--pattern-length",
		type=int,
		default=3,
		help="Pattern length (in tokens) for scenario 3 and 4.",
	)
	parser.add_argument(
		"--pattern-vocab-size",
		type=int,
		default=600,
		help="Number of most frequent patterns to keep when scenario 3/4 is enabled.",
	)
	parser.add_argument(
		"--save-signatures",
		action="store_true",
		help="Persist a small set of MinHash signatures examples per class into the output directory.",
	)
	parser.add_argument(
		"--signatures-per-class",
		type=int,
		default=3,
		help="How many signature examples to save per class when --save-signatures is set.",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="Verbosity of the console logger.",
	)
	parser.add_argument(
		"--demo",
		action="store_true",
		help="Use a small built-in dataset instead of loading Sentiment140.",
	)
	parser.add_argument(
		"--no-plots",
		action="store_true",
		help="Skip matplotlib plots (useful on headless servers).",
	)

	return parser.parse_args(argv)


def configure_logging(level: str) -> None:
	"""Initialise module-level logging."""

	logging.basicConfig(
		level=getattr(logging, level.upper(), logging.INFO),
		format="%(asctime)s | %(levelname)s | %(message)s",
	)


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
	"""Load and sample the Sentiment140 dataset (or a built-in demo sample)."""

	if args.demo:
		LOGGER.warning("Running in demo mode with a synthetic 200-row dataset.")
		data = {
			"label": ([0] * 100) + ([4] * 100),
			"text": [
				"I love sunny days and amazing friends" if i % 2 == 0 else "This is the worst day ever"
				for i in range(200)
			],
		}
		return pd.DataFrame(data)

	if not args.data_path.exists():
		raise FileNotFoundError(f"Dataset not found at {args.data_path}")

	LOGGER.info("Loading data from %s", args.data_path)
	df = pd.read_csv(
		args.data_path,
		header=None,
		encoding="ISO-8859-1",
		names=["target", "tweet_id", "date", "query", "user", "text"],
		usecols=[0, 5],
		nrows=args.max_rows if args.max_rows and args.max_rows > 0 else None,
		on_bad_lines="skip",
	)

	LOGGER.info("Loaded %d rows, dropping NaNs and duplicates", len(df))
	df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

	if args.sample_size and 0 < args.sample_size < len(df):
		df = df.sample(args.sample_size, random_state=args.random_state).reset_index(drop=True)
		LOGGER.info("Sampled %d rows", len(df))

	df = df.rename(columns={"target": "label"})
	if df["label"].nunique(dropna=True) < 2:
		msg = (
			"Loaded dataset only contains a single label. Increase --max-rows (or set it to 0)"
			" so both sentiment classes are present before sampling."
		)
		raise ValueError(msg)
	return df


def clean_text(text: str) -> str:
	"""Basic normalisation: lowercasing, URL/@user cleanup, stripping emojis."""

	text = text.lower()
	text = re.sub(r"https?://\S+", " ", text)
	text = re.sub(r"&\w+;", " ", text)
	text = re.sub(r"[^\x00-\x7F]+", " ", text)
	return text


def tokenize(text: str) -> Tuple[str, ...]:
	"""Tokenise text and drop stop words / trivial tokens."""

	if not isinstance(text, str):
		return tuple()

	cleaned = clean_text(text)
	tokens: List[str] = []
	for match in TOKEN_REGEX.finditer(cleaned):
		token = match.group().strip("'")
		if not token or token in STOP_WORDS or token.isdigit():
			continue
		tokens.append(token)

	return _dedupe_preserve_order(tokens)


def attach_tokens(df: pd.DataFrame) -> pd.DataFrame:
	"""Create a ``tokens`` column and drop rows with empty feature sets."""

	df = df.copy()
	df["tokens"] = df["text"].apply(tokenize)
	df = df[df["tokens"].map(len) > 0].reset_index(drop=True)
	return df


def apply_feature_scenario(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
	"""Attach the scenario-specific feature tokens column."""

	df = df.copy()
	initial_len = len(df)
	scenario = args.feature_scenario
	pattern_vocab: Optional[Tuple[str, ...]] = None
	if scenario in {"patterns", "all"}:
		pattern_vocab = build_pattern_vocabulary(
			df["tokens"],
			length=max(2, args.pattern_length),
			max_size=max(1, args.pattern_vocab_size),
		)
		LOGGER.info("Pattern vocabulary size=%d (top %d)", len(pattern_vocab), args.pattern_vocab_size)

	pattern_lookup = set(pattern_vocab) if pattern_vocab else None
	df["feature_tokens"] = df["tokens"].apply(
		lambda toks: select_feature_tokens(
			toks,
			scenario=scenario,
			ngram_length=max(2, args.bigram_length),
			pattern_length=max(2, args.pattern_length),
			pattern_vocab=pattern_lookup,
		)
	)
	df = df[df["feature_tokens"].map(len) > 0].reset_index(drop=True)
	retained_pct = (len(df) / initial_len) * 100 if initial_len else 0.0
	LOGGER.info("Scenario '%s' retained %d/%d samples (%.1f%%)", scenario, len(df), initial_len, retained_pct)
	return df


def build_minhash(tokens: Sequence[str], num_perm: int) -> MinHash:
	"""Generate a MinHash signature for a token sequence."""

	mh = MinHash(num_perm=num_perm)
	for token in tokens:
		mh.update(token.encode("utf-8"))
	return mh


class MinHashKNNClassifier:
	"""Simple kNN classifier backed by MinHash LSH."""

	def __init__(self, num_perm: int) -> None:
		self.num_perm = num_perm
		self.forest = MinHashLSHForest(num_perm=num_perm)
		self.key_to_tokens: Dict[str, Tuple[str, ...]] = {}
		self.key_to_label: Dict[str, int] = {}
		# keep the MinHash objects so we can export example signatures later
		self.key_to_minhash: Dict[str, MinHash] = {}

	def fit(self, tokens_list: Sequence[Sequence[str]], labels: Sequence[int]) -> float:
		"""Index training samples and return the elapsed time in seconds."""

		start = time.perf_counter()
		for idx, (tokens, label) in enumerate(zip(tokens_list, labels)):
			key = f"train_{idx}"
			mh = build_minhash(tokens, self.num_perm)
			self.forest.add(key, mh)
			self.key_to_tokens[key] = tuple(tokens)
			self.key_to_label[key] = int(label)
			self.key_to_minhash[key] = mh

		self.forest.index()
		elapsed = time.perf_counter() - start
		LOGGER.debug(
			"Indexed %d samples with num_perm=%d in %.2fs",
			len(tokens_list),
			self.num_perm,
			elapsed,
		)
		return elapsed

	def predict(
		self, tokens: Sequence[str], k_neighbors: int, fallback_label: int
	) -> Tuple[int, float, int]:
		"""Return (label, query_time_sec, candidate_count)."""

		mh = build_minhash(tokens, self.num_perm)
		start = time.perf_counter()
		neighbor_keys = self.forest.query(mh, k_neighbors)
		query_time = time.perf_counter() - start

		if not neighbor_keys:
			return fallback_label, query_time, 0

		votes: Dict[int, float] = defaultdict(float)
		target_set = set(tokens)
		for key in neighbor_keys:
			candidate_tokens = self.key_to_tokens.get(key)
			if candidate_tokens is None:
				continue
			label = self.key_to_label[key]
			candidate_set = set(candidate_tokens)
			union = len(target_set | candidate_set)
			score = len(target_set & candidate_set) / union if union else 0.0
			votes[label] += score if score > 0 else 1e-6

		if not votes:
			return fallback_label, query_time, len(neighbor_keys)

		best_label = max(votes.items(), key=lambda item: (item[1], -item[0]))[0]
		return best_label, query_time, len(neighbor_keys)


def run_single_experiment(
	classifier: MinHashKNNClassifier,
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
	config: ExperimentConfig,
	output_dir: Optional[Path] = None,
	signature_examples: int = 0,
) -> ExperimentResult:
	"""Fit the classifier and evaluate a specific (num_perm, k) pair."""

	LOGGER.info(
		"Evaluating numHashTables=%d | k=%d",
		config.num_perm,
		config.k_neighbors,
	)

	training_time = classifier.fit(train_df["feature_tokens"].tolist(), train_df["label"].tolist())

	# Export signature examples per class when requested
	if signature_examples and output_dir is not None:
		try:
			save_signature_examples(classifier, output_dir, n_examples=signature_examples)
		except Exception as exc:  # keep experiments resilient to export errors
			LOGGER.warning("Failed to save signature examples: %s", exc)
	fallback_label = int(train_df["label"].mode(dropna=False).iloc[0])

	correct = 0
	query_times: List[float] = []
	candidate_counts: List[int] = []

	for tokens, label in zip(test_df["feature_tokens"], test_df["label"]):
		prediction, q_time, n_candidates = classifier.predict(tokens, config.k_neighbors, fallback_label)
		query_times.append(q_time)
		candidate_counts.append(n_candidates)
		if prediction == int(label):
			correct += 1

	accuracy = correct / len(test_df)
	avg_query_time_ms = float(np.mean(query_times) * 1000.0)
	avg_candidates = float(np.mean(candidate_counts))
	throughput = train_df.shape[0] / training_time if training_time > 0 else float("inf")

	return ExperimentResult(
		num_perm=config.num_perm,
		k_neighbors=config.k_neighbors,
		accuracy=accuracy,
		avg_query_time_ms=avg_query_time_ms,
		avg_candidates=avg_candidates,
		training_time_sec=training_time,
		throughput_samples_per_sec=throughput,
		train_size=train_df.shape[0],
		test_size=test_df.shape[0],
	)


def run_experiments(args: argparse.Namespace) -> List[ExperimentResult]:
	"""Coordinate the full experiment grid."""

	df = attach_tokens(load_dataset(args))
	df = apply_feature_scenario(df, args)
	LOGGER.info("Data prepared: %d usable tweets", len(df))

	label_mapping = {4: 1, 0: 0}
	df["label"] = df["label"].map(label_mapping).fillna(df["label"]).astype(int)

	stratify = df["label"] if df["label"].nunique() > 1 else None
	train_df, test_df = train_test_split(
		df,
		test_size=args.test_fraction,
		random_state=args.random_state,
		stratify=stratify,
	)

	results: List[ExperimentResult] = []
	for num_perm in args.num_hash_tables:
		for k_neighbors in args.k_values:
			config = ExperimentConfig(num_perm=num_perm, k_neighbors=k_neighbors)
			classifier = MinHashKNNClassifier(num_perm=num_perm)
			# decide whether to export signature examples for this run
			sig_examples = args.signatures_per_class if getattr(args, "save_signatures", False) else 0
			result = run_single_experiment(
				classifier,
				train_df,
				test_df,
				config,
				output_dir=args.output_dir,
				signature_examples=sig_examples,
			)
			results.append(result)

	return results


def results_to_dataframe(results: Sequence[ExperimentResult]) -> pd.DataFrame:
	"""Convert dataclasses to a tidy pandas DataFrame."""

	return pd.DataFrame([
		{
			"num_hash_tables": r.num_perm,
			"k": r.k_neighbors,
			"accuracy": r.accuracy,
			"avg_query_time_ms": r.avg_query_time_ms,
			"avg_candidates": r.avg_candidates,
			"training_time_sec": r.training_time_sec,
			"throughput_samples_per_sec": r.throughput_samples_per_sec,
			"train_size": r.train_size,
			"test_size": r.test_size,
		}
		for r in results
	])


def save_signature_examples(classifier: MinHashKNNClassifier, output_dir: Path, n_examples: int = 3) -> None:
	"""Export up to ``n_examples`` MinHash signatures per class as JSON.

	The output file is written to ``<output_dir>/signatures/signature_examples.json``.
	Each entry contains: key, tokens (list) and signature (list of integers).
	"""

	if n_examples <= 0:
		return

	output_dir.mkdir(parents=True, exist_ok=True)
	sig_dir = output_dir / "signatures"
	sig_dir.mkdir(parents=True, exist_ok=True)

	examples_by_label: Dict[int, List[Dict[str, object]]] = {}
	for key, label in classifier.key_to_label.items():
		mh = classifier.key_to_minhash.get(key)
		if mh is None:
			continue
		tokens = list(classifier.key_to_tokens.get(key, ()))
		# datasketch.MinHash exposes `hashvalues` as a numpy array; convert to plain ints
		sig_values = getattr(mh, "hashvalues", None)
		if sig_values is None:
			sig_list = []
		else:
			sig_list = [int(x) for x in sig_values.tolist()]

		examples_by_label.setdefault(int(label), []).append({
			"key": key,
			"tokens": tokens,
			"signature": sig_list,
		})

	# truncate to requested number per class and write JSON
	for label, items in examples_by_label.items():
		examples_by_label[label] = items[:max(1, int(n_examples))]

	out_path = sig_dir / "signature_examples.json"
	with out_path.open("w", encoding="utf-8") as fh:
		json.dump({str(k): v for k, v in examples_by_label.items()}, fh, ensure_ascii=False, indent=2)

	LOGGER.info("Saved signature examples (up to %d per class) to %s", n_examples, out_path)


def ensure_output_dir(directory: Path) -> None:
	"""Create the output directory tree if necessary."""

	directory.mkdir(parents=True, exist_ok=True)


def save_tables(df: pd.DataFrame, output_dir: Path) -> None:
	"""Persist CSV and Markdown tables."""

	csv_path = output_dir / "lsh_results.csv"
	md_path = output_dir / "lsh_results.md"
	df_sorted = df.sort_values(["num_hash_tables", "k"]).reset_index(drop=True)

	df_sorted.to_csv(csv_path, index=False)
	LOGGER.info("Saved raw CSV table to %s", csv_path)

	md_lines = ["| numHashTables | k | accuracy | avg_query_time_ms | avg_candidates | training_time_sec | throughput_samples_per_sec |",
				"| --- | --- | --- | --- | --- | --- | --- |"]
	for row in df_sorted.itertuples(index=False):
		md_lines.append(
			f"| {row.num_hash_tables} | {row.k} | {row.accuracy:.4f} | {row.avg_query_time_ms:.2f} | "
			f"{row.avg_candidates:.1f} | {row.training_time_sec:.2f} | {row.throughput_samples_per_sec:.2f} |"
		)

	md_path.write_text("\n".join(md_lines), encoding="utf-8")
	LOGGER.info("Saved Markdown table to %s", md_path)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
	"""Plot ``metric`` against k for each numHashTables value."""

	fig, ax = plt.subplots(figsize=(7, 4))
	for num_hash_tables, group in df.groupby("num_hash_tables"):
		grouped = group.sort_values("k")
		ax.plot(grouped["k"], grouped[metric], marker="o", label=f"numHashTables={num_hash_tables}")
	ax.set_xlabel("k (nearest neighbours)")
	ax.set_ylabel(ylabel)
	ax.set_title(f"{ylabel} vs k")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	plt.close(fig)
	LOGGER.info("Saved plot to %s", output_path)


def generate_plots(df: pd.DataFrame, output_dir: Path, skip_plots: bool) -> None:
	"""Create accuracy and query-time plots unless disabled."""

	if skip_plots:
		LOGGER.warning("Plot generation skipped as requested.")
		return

	plot_metric(df, "accuracy", "Accuracy", output_dir / "accuracy_vs_k.png")
	plot_metric(df, "avg_query_time_ms", "Average query time (ms)", output_dir / "query_time_vs_k.png")


def print_summary(df: pd.DataFrame) -> None:
	"""Pretty-print the top-performing configuration."""

	best_row = df.sort_values("accuracy", ascending=False).iloc[0]
	LOGGER.info(
		"Best config: numHashTables=%d | k=%d | accuracy=%.4f | avg_query_time=%.2f ms",
		best_row.num_hash_tables,
		best_row.k,
		best_row.accuracy,
		best_row.avg_query_time_ms,
	)


def main(argv: Optional[Sequence[str]] = None) -> None:
	args = parse_args(argv)
	configure_logging(args.log_level)

	ensure_output_dir(args.output_dir)
	results = run_experiments(args)
	df = results_to_dataframe(results)

	save_tables(df, args.output_dir)
	generate_plots(df, args.output_dir, skip_plots=args.no_plots)
	print_summary(df)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		LOGGER.error("Execution interrupted by user")
		sys.exit(130)
