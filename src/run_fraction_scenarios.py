

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "training.1600000.processed.noemoticon.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "fraction_runs"
DEFAULT_EXPERIMENT_SCRIPT = REPO_ROOT / "src" / "experiment_lsh.py"
DEFAULT_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)

SCENARIOS = (
    ("scenario1_unigram", "unigram", "Scénario 1 – Unigrammes"),
    ("scenario2_bigram", "bigram", "Scénario 2 – Bigrams"),
    ("scenario3_patterns", "patterns", "Scénario 3 – Motifs fréquents"),
    ("scenario4_all", "all", "Scénario 4 – Combiné"),
)


@dataclass
class SummaryRecord:
    scenario_label: str
    scenario: str
    scenario_display: str
    fraction_requested: float
    fraction_actual: float
    sample_size: int
    num_hash_tables: int
    k: int
    accuracy: float
    training_time_sec: float
    evaluation_time_sec: float
    total_time_sec: float
    output_dir: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSH scenarios across dataset fractions.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Chemin complet vers le CSV Sentiment140.",
    )
    parser.add_argument(
        "--experiment-script",
        type=Path,
        default=DEFAULT_EXPERIMENT_SCRIPT,
        help="Chemin vers experiment_lsh.py (permet d'utiliser un fork).",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Interpréteur Python à utiliser pour lancer experiment_lsh.py.",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="*",
        default=DEFAULT_FRACTIONS,
        help="Fractions du sous-échantillon max (base_sample_size) à exécuter.",
    )
    parser.add_argument(
        "--base-sample-size",
        type=int,
        default=50_000,
        help="Nombre maximum d'exemples à considérer avant de calculer les fractions.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction dédiée au test (transmise à experiment_lsh.py).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Nombre maximum de lignes à lire avant l'échantillonnage (0 = tout le fichier).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire racine où stocker les sous-résultats et graphiques.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        choices=[label for label, _, _ in SCENARIOS],
        default=[label for label, _, _ in SCENARIOS],
        help="Limiter l'exécution à un sous-ensemble des scénarios.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Ne pas relancer un run si le CSV existe déjà pour ce couple scénario/fraction.",
    )
    parser.add_argument(
        "experiment_args",
        nargs=argparse.REMAINDER,
        help="Arguments additionnels à passer tels quels à experiment_lsh.py (après --).",
    )
    return parser.parse_args(argv)


def ensure_dependencies(args: argparse.Namespace) -> None:
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {args.data_path}")
    if not args.experiment_script.exists():
        raise FileNotFoundError(f"Script experiment_lsh.py introuvable: {args.experiment_script}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def normalize_experiment_args(raw: Sequence[str] | None) -> List[str]:
    if not raw:
        return []
    cleaned = list(raw)
    if cleaned and cleaned[0] == "--":
        cleaned = cleaned[1:]
    return cleaned


def format_fraction_label(fraction: float) -> str:
    return f"fraction_{int(round(fraction * 100)):02d}"


def run_single_experiment(
    args: argparse.Namespace,
    scenario_label: str,
    scenario_flag: str,
    fraction: float,
    output_dir: Path,
) -> None:
    sample_size = max(1, int(round(args.base_sample_size * fraction)))
    fraction_display = sample_size / args.base_sample_size
    target_dir = output_dir / scenario_label / format_fraction_label(fraction)
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / "lsh_results.csv"
    if args.keep_existing and csv_path.exists():
        print(f"[SKIP] {scenario_label} fraction {fraction_display:.2f} (CSV déjà présent)")
        return

    max_rows_arg = args.max_rows if args.max_rows and args.max_rows > 0 else 0

    cmd: List[str] = [
        args.python_bin,
        str(args.experiment_script),
        "--data-path",
        str(args.data_path),
        "--output-dir",
        str(target_dir),
        "--feature-scenario",
        scenario_flag,
    "--max-rows",
    str(max_rows_arg),
        "--sample-size",
        str(sample_size),
        "--test-fraction",
        str(args.test_fraction),
        "--random-state",
        "42",
    ]

    if args.experiment_args:
        cmd.extend(args.experiment_args)

    print("==============================================================")
    print(f"Scénario {scenario_label} | fraction demandée {fraction:.2f} | échantillon {sample_size}")
    print(f"Sortie: {target_dir}")
    print("==============================================================")
    subprocess.run(cmd, check=True)


def parse_best_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Le fichier {csv_path} est vide.")
    df = df.copy()
    if "evaluation_time_sec" not in df.columns:
        df["evaluation_time_sec"] = 0.0
    df["total_time_sec"] = df["training_time_sec"] + df["evaluation_time_sec"]
    df = df.sort_values(by=["accuracy", "total_time_sec"], ascending=[False, True])
    return df.iloc[0]


def collect_summaries(
    base_output: Path,
    selected_scenarios: set[str],
    fractions: Iterable[float],
    base_sample_size: int,
) -> List[SummaryRecord]:
    records: List[SummaryRecord] = []
    for scenario_label, scenario_flag, display in SCENARIOS:
        if scenario_label not in selected_scenarios:
            continue
        for fraction in fractions:
            sample_size = max(1, int(round(base_sample_size * fraction)))
            subdir = base_output / scenario_label / format_fraction_label(fraction)
            csv_path = subdir / "lsh_results.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Résultat manquant pour {scenario_label} fraction {fraction:.2f}: {csv_path}"
                )
            best = parse_best_row(csv_path)
            fraction_actual = sample_size / base_sample_size
            records.append(
                SummaryRecord(
                    scenario_label=scenario_label,
                    scenario=scenario_flag,
                    scenario_display=display,
                    fraction_requested=fraction,
                    fraction_actual=fraction_actual,
                    sample_size=sample_size,
                    num_hash_tables=int(best["num_hash_tables"]),
                    k=int(best["k"]),
                    accuracy=float(best["accuracy"]),
                    training_time_sec=float(best["training_time_sec"]),
                    evaluation_time_sec=float(best["evaluation_time_sec"]),
                    total_time_sec=float(best["total_time_sec"]),
                    output_dir=subdir,
                )
            )
    return records


def plot_execution_time(df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    for scenario_label, _, display in SCENARIOS:
        subset = df[df["scenario_label"] == scenario_label].sort_values("fraction_actual")
        if subset.empty:
            continue
        plt.plot(
            subset["fraction_actual"],
            subset["total_time_sec"],
            marker="o",
            label=display,
        )
    plt.xlabel("Fraction du sous-échantillon (sur 10 000 tweets)")
    plt.ylabel("Temps total (s)")
    plt.title("Temps d'exécution vs fraction")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    path = output_dir / "execution_time_vs_fraction.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_training_histogram(df: pd.DataFrame, output_dir: Path) -> Path:
    fractions = sorted(df["fraction_actual"].unique())
    if not fractions:
        raise ValueError("Aucune fraction disponible pour générer l'histogramme.")
    x = np.arange(len(fractions))
    width = 0.8 / len([label for label, _, _ in SCENARIOS])

    plt.figure(figsize=(9, 5))
    for idx, (scenario_label, _, display) in enumerate(SCENARIOS):
        subset = df[df["scenario_label"] == scenario_label]
        if subset.empty:
            continue
        y = []
        for frac in fractions:
            row = subset[np.isclose(subset["fraction_actual"], frac)]
            y.append(float(row["training_time_sec"].iloc[0]) if not row.empty else 0.0)
        offsets = x - 0.4 + width / 2 + idx * width
        plt.bar(offsets, y, width=width, label=display)

    plt.xticks(x, [f"{frac:.2f}" for frac in fractions])
    plt.xlabel("Fraction du sous-échantillon (sur 10 000 tweets)")
    plt.ylabel("Temps d'entraînement (s)")
    plt.title("Temps d'entraînement par scénario et fraction")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    path = output_dir / "training_time_histogram.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_summary(records: List[SummaryRecord], output_dir: Path) -> Path:
    df = pd.DataFrame([r.__dict__ for r in records])
    csv_path = output_dir / "fraction_summary.csv"
    md_path = output_dir / "fraction_summary.md"
    df.to_csv(csv_path, index=False)
    df_md = df.copy()
    df_md["fraction_requested"] = df_md["fraction_requested"].map(lambda x: f"{x:.2f}")
    df_md["fraction_actual"] = df_md["fraction_actual"].map(lambda x: f"{x:.4f}")
    md_path.write_text(dataframe_to_markdown(df_md), encoding="utf-8")
    return csv_path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = []
    for row in df.itertuples(index=False):
        values = [str(value) for value in row]
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header_line, separator_line, *body_lines])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.experiment_args = normalize_experiment_args(args.experiment_args)
    ensure_dependencies(args)

    fractions = sorted({max(0.05, min(1.0, f)) for f in args.fractions})
    selected_scenarios = set(args.scenarios)

    for scenario_label, scenario_flag, _ in SCENARIOS:
        if scenario_label not in selected_scenarios:
            continue
        for fraction in fractions:
            run_single_experiment(args, scenario_label, scenario_flag, fraction, args.output_dir)

    records = collect_summaries(args.output_dir, selected_scenarios, fractions, args.base_sample_size)
    summary_csv = save_summary(records, args.output_dir)
    summary_df = pd.read_csv(summary_csv)
    exec_plot = plot_execution_time(summary_df, args.output_dir)
    hist_plot = plot_training_histogram(summary_df, args.output_dir)

    print("==============================================================")
    print("RÉCAPITULATIF DES FRACTIONS")
    print(summary_df)
    print(f"Résumé CSV : {summary_csv}")
    print(f"Courbe temps vs fraction : {exec_plot}")
    print(f"Histogramme training : {hist_plot}")
    print("==============================================================")


if __name__ == "__main__":
    main()
