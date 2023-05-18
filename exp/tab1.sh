#! /bin/bash

python3 ../main.py src_lang="en" tgt_lang="de" device="cpu" dataset.name="wmt" dataset.version="14" model.model_name="Helsinki-NLP/opus-mt-en-de"
python3 ../main.py src_lang="de" tgt_lang="en" device="cpu" dataset.name="wmt" dataset.version="14" model.model_name="Helsinki-NLP/opus-mt-en-de"
python3 ../main.py src_lang="en" tgt_lang="de" device="cpu" dataset.name="wmt" dataset.version="14" model.model_name="teelinsan/opus-mt-eng-deu"
python3 ../main.py src_lang="de" tgt_lang="en" device="cpu" dataset.name="wmt" dataset.version="14" model.model_name="teelinsan/opus-mt-eng-deu"
python3 ../main.py src_lang="ro" tgt_lang="en" device="cpu" dataset.name="wmt" dataset.version="16" model.model_name="Helsinki-NLP/opus-mt-roa-en"
python3 ../main.py src_lang="en" tgt_lang="ro" device="cpu" dataset.name="wmt" dataset.version="16" model.model_name="Helsinki-NLP/opus-mt-roa-en"

python3 ../main.py src_lang="en" tgt_lang="de" device="cpu" dataset.name="wmt" dataset.version="14" task="beam_search" model.model_name="teelinsan/opus-mt-eng-deu"
python3 ../main.py src_lang="de" tgt_lang="en" device="cpu" dataset.name="wmt" dataset.version="14" task="beam_search"
python3 ../main.py src_lang="en" tgt_lang="ro" device="cpu" dataset.name="wmt" dataset.version="16" task="beam_search"
python3 ../main.py src_lang="ro" tgt_lang="en" device="cpu" dataset.name="wmt" dataset.version="16" task="beam_search" model.model_name="Helsinki-NLP/opus-mt-roa-en"


python3 ../main.py src_lang="en" tgt_lang="de" device="cuda" dataset.name="wmt" dataset.version="14" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="de" tgt_lang="en" device="cuda" dataset.name="wmt" dataset.version="14" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="ro" tgt_lang="en" device="cuda" dataset.name="wmt" dataset.version="16" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="en" tgt_lang="ro" device="cuda" dataset.name="wmt" dataset.version="16" model.model_name="facebook/mbart-large-50-many-to-many-mmt"

python3 ../main.py src_lang="en" tgt_lang="de" device="cuda" dataset.name="wmt" dataset.version="14" task="beam_search" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="de" tgt_lang="en" device="cuda" dataset.name="wmt" dataset.version="14" task="beam_search" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="en" tgt_lang="ro" device="cuda" dataset.name="wmt" dataset.version="16" task="beam_search" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="ro" tgt_lang="en" device="cuda" dataset.name="wmt" dataset.version="16" task="beam_search" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
