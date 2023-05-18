#! /bin/bash

python3 ../main.py src_lang="en" tgt_lang="fi" device="cuda" dataset.name="wmt" dataset.version="17" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="fi" tgt_lang="en" device="cuda" dataset.name="wmt" dataset.version="17" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="en" tgt_lang="hi" device="cuda" dataset.name="ittb" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="hi" tgt_lang="en" device="cuda" dataset.name="ittb" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="en" tgt_lang="vi" device="cuda" dataset.name="iwslt" dataset.version="15" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="vi" tgt_lang="en" device="cuda" dataset.name="iwslt" dataset.version="15" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="en" tgt_lang="it" device="cuda" dataset.name="flores" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="it" tgt_lang="en" device="cuda" dataset.name="flores" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="en" tgt_lang="fr" device="cuda" dataset.name="flores" model.model_name="facebook/mbart-large-50-many-to-many-mmt"
python3 ../main.py src_lang="fr" tgt_lang="en" device="cuda" dataset.name="flores" model.model_name="facebook/mbart-large-50-many-to-many-mmt"



python3 ../main.py src_lang="en" tgt_lang="fi" device="cuda" dataset.name="wmt" dataset.version="17" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"
python3 ../main.py src_lang="fi" tgt_lang="en" device="cuda" dataset.name="wmt" dataset.version="17" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"
python3 ../main.py src_lang="en" tgt_lang="hi" device="cuda" dataset.name="ittb" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"
python3 ../main.py src_lang="hi" tgt_lang="en" device="cuda" dataset.name="ittb" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"
python3 ../main.py src_lang="en" tgt_lang="vi" device="cuda" dataset.name="iwslt" dataset.version="15" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"
python3 ../main.py src_lang="vi" tgt_lang="en" device="cuda" dataset.name="iwslt" dataset.version="15" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"
python3 ../main.py src_lang="en" tgt_lang="it" device="cuda" dataset.name="flores" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"
python3 ../main.py src_lang="it" tgt_lang="en" device="cuda" dataset.name="flores" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"
python3 ../main.py src_lang="en" tgt_lang="fr" device="cuda" dataset.name="flores" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"
python3 ../main.py src_lang="fr" tgt_lang="en" device="cuda" dataset.name="flores" model.model_name="facebook/mbart-large-50-many-to-many-mmt" task="beam_search"

