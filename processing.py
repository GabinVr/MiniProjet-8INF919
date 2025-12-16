#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessing des tweets Sentiment140 avec PySpark.

Étapes :
1. Chargement du CSV
2. Nettoyage du texte
3. Tokenisation + suppression des stopwords
4. Création des 1-gram (mots) et 2-gram (paires de mots)
5. HashingTF pour obtenir un vecteur de features
6. Sauvegarde en parquet

Utilisation (exemple) :
spark-submit preprocessing.py \
  --input /chemin/vers/sentiment140.csv \
  --output /chemin/vers/sortie_parquet \
  --numFeatures 262144
"""

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    NGram,
    HashingTF,
    VectorAssembler,
)

# ---------------------------------------------------------------------
# 1. Fonction de nettoyage du texte (DataFrame -> nouvelle colonne)
# ---------------------------------------------------------------------

def clean_text_column(df, text_col="text", output_col="clean_text"):
    """
    Nettoie la colonne textuelle en plusieurs étapes :
    - met tout en minuscules
    - enlève les URLs
    - enlève la ponctuation / caractères bizarres
    - enlève les espaces en trop
    """

    # 1) mettre en minuscules
    col = F.lower(F.col(text_col))

    # 2) enlever les URLs (http..., https..., www...)
    col = F.regexp_replace(col, r"http\S+|www\.\S+", " ")

    # 3) enlever tout ce qui n'est pas lettre ou espace
    #    (on garde juste a-z et les espaces)
    col = F.regexp_replace(col, r"[^a-z\s]", " ")

    # 4) remplacer les espaces multiples par un seul
    col = F.regexp_replace(col, r"\s+", " ")

    # 5) trim : enlever espace au début et à la fin
    col = F.trim(col)

    return df.withColumn(output_col, col)


# ---------------------------------------------------------------------
# 2. Construction du pipeline Spark ML
# ---------------------------------------------------------------------

def build_pipeline(num_features=262144):
    """
    Construit un Pipeline Spark qui fait :
    - Tokenizer : texte -> liste de mots
    - StopWordsRemover : enlève les mots très fréquents (the, a, is, ...)
    - NGram(n=2) : crée les 2-gram (paires de mots)
    - HashingTF sur les 1-gram
    - HashingTF sur les 2-gram
    - VectorAssembler : concatène les deux en une colonne 'features'
    """

    # 1) Découper le texte en liste de mots
    tokenizer = Tokenizer(
        inputCol="clean_text",
        outputCol="tokens"
    )

    # 2) Enlever les stopwords (mots très fréquents et peu informatifs)
    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="unigrams"  # ce seront nos 1-gram
    )

    # 3) Créer les 2-gram (paires de mots)
    bigrammer = NGram(
        n=2,
        inputCol="unigrams",
        outputCol="bigrams"
    )

    # 4) Transformer les 1-gram en vecteurs via HashingTF
    hashing_unigrams = HashingTF(
        numFeatures=num_features,
        inputCol="unigrams",
        outputCol="unigram_features"
    )

    # 5) Transformer les 2-gram en vecteurs via HashingTF
    hashing_bigrams = HashingTF(
        numFeatures=num_features,
        inputCol="bigrams",
        outputCol="bigram_features"
    )

    # 6) Assembler les deux types de features dans une seule colonne "features"
    assembler = VectorAssembler(
        inputCols=["unigram_features", "bigram_features"],
        outputCol="features"
    )

    pipeline = Pipeline(stages=[
        tokenizer,
        remover,
        bigrammer,
        hashing_unigrams,
        hashing_bigrams,
        assembler
    ])

    return pipeline


# ---------------------------------------------------------------------
# 3. Fonction principale
# ---------------------------------------------------------------------

def main(args):
    # 1) Créer la session Spark
    spark = (
        SparkSession.builder
        .appName("Sentiment140_Preprocessing")
        .getOrCreate()
    )

    # Pour éviter trop de logs
    spark.sparkContext.setLogLevel("WARN")

    # 2) Charger le CSV
    # Si ton fichier n'a PAS d'entête, mets header=False et adapte les noms.
    df = (
        spark.read
        .option("header", True)      # True si le CSV a un header
        .option("inferSchema", True) # Spark devine les types (int, string...)
        .csv(args.input)
    )

    # 3) Garder seulement les colonnes utiles (ici label + texte)
    #    Adapte 'target' et 'text' si tes colonnes ont d'autres noms.
    df = df.select(
        F.col(args.label_col).alias("label"),
        F.col(args.text_col).cast(StringType()).alias("text")
    ).dropna(subset=["label", "text"])

    # 4) Nettoyer le texte
    df_clean = clean_text_column(df, text_col="text", output_col="clean_text")

    # 5) Construire le pipeline
    pipeline = build_pipeline(num_features=args.numFeatures)

    # 6) Entraîner le pipeline sur les données (fit) puis transformer
    model = pipeline.fit(df_clean)
    df_final = model.transform(df_clean)

    # 7) Garder les colonnes finales qui serviront aux modèles
    #    'features' : vecteur sparse
    #    'label'    : 0 / 2 / 4 (ou autre)
    df_output = df_final.select("label", "features")

    # 8) Sauvegarder en parquet
    (
        df_output
        .write
        .mode("overwrite")
        .parquet(args.output)
    )

    print("✅ Préprocessing terminé, données sauvegardées dans :", args.output)

    spark.stop()


# ---------------------------------------------------------------------
# 4. Point d'entrée du script
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Sentiment140 avec PySpark")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="./train.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="./output"
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="sentiment",
    )
    parser.add_argument(
        "--numFeatures",
        type=int,
        default=262144,  # 2^18, taille classique pour HashingTF
        help="Taille du vecteur HashingTF (dimension des features)"
    )

    args = parser.parse_args()
    main(args)