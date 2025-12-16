#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --time=00:40:00
#SBATCH --nodes=4
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=sent140_lsh_neutral

set -euo pipefail

module load StdEnv/2023
module load scipy-stack
module load spark/3.5.6

echo "Modules chargés"

mkdir -p logs

export MKL_NUM_THREADS=1
export SPARK_IDENT_STRING=$SLURM_JOBID
export SPARK_WORKER_DIR=$SLURM_TMPDIR/spark-work
export SPARK_LOG_DIR=$SLURM_TMPDIR/spark-logs
mkdir -p "$SPARK_WORKER_DIR" "$SPARK_LOG_DIR"

export SLURM_SPARK_MEM=$(printf "%.0f" $((${SLURM_MEM_PER_NODE} * 95 / 100)))

cleanup () {
  echo "=== Nettoyage Spark ==="
  stop-master.sh >/dev/null 2>&1 || true
  if [[ -n "${slaves_pid:-}" ]]; then
    kill "${slaves_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

start-master.sh
sleep 5

MASTER_URL=$(grep -Po '(?=spark://).*' "$SPARK_LOG_DIR"/spark-"${SPARK_IDENT_STRING}"-org.apache.spark.deploy.master*.out | tail -n 1)
echo "MASTER_URL=${MASTER_URL}"

NWORKERS=$((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES - 1))
echo "Démarrage de ${NWORKERS} workers"

SPARK_NO_DAEMONIZE=1 srun -n "${NWORKERS}" -N "${NWORKERS}" --label \
  --output="$SPARK_LOG_DIR/spark-${SLURM_JOBID}-workers.out" \
  start-slave.sh -m "${SLURM_SPARK_MEM}"M -c "${SLURM_CPUS_PER_TASK}" "${MASTER_URL}" &

slaves_pid=$!

sleep 10

echo "==========================================================================="
echo "EXÉCUTION MinHashLSH sur Sentiment140 (AVEC NEUTRE)"
echo "==========================================================================="

echo ""
echo "-------------------------------------------------------------------"
echo "On utilise l'environement virtuel Python"
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
echo "-------------------------------------------------------------------"
export PYSPARK_PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"

cd "$SLURM_SUBMIT_DIR"

SCRIPT_PY="$SLURM_SUBMIT_DIR/src/experiment_lsh_spark.py"
# Utilise le dataset Kaggle (avec neutre) situé dans src/data
DATA_PATH="$SLURM_SUBMIT_DIR/src/data/train.csv"
TEST_PATH="$SLURM_SUBMIT_DIR/src/data/test.csv"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/reports/lsh_spark_neutral"

mkdir -p "$OUTPUT_DIR"

srun -n 1 -N 1 spark-submit \
  --master "${MASTER_URL}" \
  --deploy-mode client \
  --driver-memory "${SLURM_SPARK_MEM}M" \
  --executor-memory "${SLURM_SPARK_MEM}M" \
  --conf spark.sql.shuffle.partitions=200 \
  --conf spark.pyspark.python="${PYSPARK_PYTHON}" \
  --conf spark.pyspark.driver.python="${PYSPARK_PYTHON}" \
  --conf spark.default.parallelism=100 \
  --conf spark.executor.cores="${SLURM_CPUS_PER_TASK}" \
  --conf spark.cores.max="$((NWORKERS * SLURM_CPUS_PER_TASK))" \
  "${SCRIPT_PY}" \
    --data-path "${DATA_PATH}" \
    --test-path "${TEST_PATH}" \
    --dataset-format tweetextraction \
    --output-dir "${OUTPUT_DIR}" \
    --samples-per-class 6000 \
    --test-fraction 0.2 \
    --random-state 42 \
    --num-hash-tables 128 250 \
    --k-values 50 100 150 200 \
    --num-features 32768 \
    --similarity-threshold 0.9 \
    --log-level INFO \
    --use-neutral \
    --scenario 2


echo "==========================================================================="
echo "JOB TERMINÉ"
echo "==========================================================================="