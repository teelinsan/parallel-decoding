import csv
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.nn.modules import linear
import datasets

def makedirs(path):
    if path.endswith((".tsv", ".csv", ".txt")):
        path = "/".join(path.split("/")[:-1])

    if not os.path.exists(path):
        os.makedirs(path)

def check_zero_division(a, b):
    return "na" if b == 0 else round(a / b, 3)

def get_logits_preprocessor(model, input_ids, eos_token_id):
    logits_preprocessor = model._get_logits_processor(
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        input_ids_seq_length=1,
        encoder_input_ids=input_ids,
        bad_words_ids=None,
        min_length=0,
        max_length=model.config.max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        prefix_allowed_tokens_fn=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_penalty=None,
        remove_invalid_values=None,
        exponential_decay_length_penalty=None,
        logits_processor=[],
        renormalize_logits=None
    )
    return logits_preprocessor



def retrieve_model_name(model_name):
    if "opus" in model_name:
        return "opus"
    if "mbart" in model_name:
        if "50" in model_name:
            return "mbart_50"
        return "mbart"


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def retrieve_map_languages_flores(lan):
    lang_map = {
        "ab": "Abkhaz",
        "aa": "Afar",
        "af": "Afrikaans",
        "ak": "Akan",
        "sq": "Albanian",
        "am": "Amharic",
        "ar": "Arabic",
        "an": "Aragonese",
        "hy": "Armenian",
        "as": "Assamese",
        "av": "Avaric",
        "ae": "Avestan",
        "ay": "Aymara",
        "az": "Azerbaijani",
        "bm": "Bambara",
        "ba": "Bashkir",
        "eu": "Basque",
        "be": "Belarusian",
        "bn": "Bengali",
        "bh": "Bihari",
        "bi": "Bislama",
        "bs": "Bosnian",
        "br": "Breton",
        "bg": "Bulgarian",
        "my": "Burmese",
        "ca": "Catalan",
        "ch": "Chamorro",
        "ce": "Chechen",
        "ny": "Chichewa",
        "zh": "Chinese",
        "cv": "Chuvash",
        "kw": "Cornish",
        "co": "Corsican",
        "cr": "Cree",
        "hr": "Croatian",
        "cs": "Czech",
        "da": "Danish",
        "dv": "Divehi",
        "nl": "Dutch",
        "dz": "Dzongkha",
        "en": "English",
        "eo": "Esperanto",
        "et": "Estonian",
        "ee": "Ewe",
        "fo": "Faroese",
        "fj": "Fijian",
        "fi": "Finnish",
        "fr": "Franch",
        "ff": "Fula",
        "gl": "Galician",
        "ka": "Georgian",
        "de": "German",
        "el": "Greek",
        "gn": "Guaraní",
        "gu": "Gujarati",
        "ht": "Haitian",
        "ha": "Hausa",
        "he": "Hebrew",
        "hz": "Herero",
        "hi": "Hindi",
        "ho": "Hiri Motu",
        "hu": "Hungarian",
        "ia": "Interlingua",
        "id": "Indonesian",
        "ie": "Interlingue",
        "ga": "Irish",
        "ig": "Igbo",
        "ik": "Inupiaq",
        "io": "Ido",
        "is": "Icelandic",
        "it": "Italian",
        "iu": "Inuktitut",
        "ja": "Japanese",
        "jv": "Javanese",
        "kl": "Kalaallisut",
        "kn": "Kannada",
        "kr": "Kanuri",
        "ks": "Kashmiri",
        "kk": "Kazakh",
        "km": "Khmer",
        "ki": "Kikuyu",
        "rw": "Kinyarwanda",
        "ky": "Kirghiz",
        "kv": "Komi",
        "kg": "Kongo",
        "ko": "Korean",
        "ku": "Kurdish",
        "kj": "Kwanyama",
        "la": "Latin",
        "lb": "Luxembourgish",
        "lg": "Luganda",
        "li": "Limburgish",
        "ln": "Lingala",
        "lo": "Lao",
        "lt": "Lithuanian",
        "lu": "Luba-Katanga",
        "lv": "Latvian",
        "gv": "Manx",
        "mk": "Macedonian",
        "mg": "Malagasy",
        "ms": "Malay",
        "ml": "Malayalam",
        "mt": "Maltese",
        "mi": "Māori",
        "mr": "Marathi",
        "mh": "Marshallese",
        "mn": "Mongolian",
        "na": "Nauru",
        "nv": "Navajo",
        "nb": "Norwegian Bokmål",
        "nd": "North Ndebele",
        "ne": "Nepali",
        "ng": "Ndonga",
        "nn": "Norwegian Nynorsk",
        "no": "Norwegian",
        "ii": "Nuosu",
        "nr": "South Ndebele",
        "oc": "Occitan",
        "oj": "Ojibwe",
        "cu": "Old Church Slavonic",
        "om": "Oromo",
        "or": "Oriya",
        "os": "Ossetian",
        "pa": "Panjabi",
        "pi": "Pāli",
        "fa": "Persian",
        "pl": "Polish",
        "ps": "Pashto",
        "pt": "Portuguese",
        "qu": "Quechua",
        "rm": "Romansh",
        "rn": "Kirundi",
        "ro": "Romanian",
        "ru": "Russian",
        "sa": "Sanskrit",
        "sc": "Sardinian",
        "sd": "Sindhi",
        "se": "Northern Sami",
        "sm": "Samoan",
        "sg": "Sango",
        "sr": "Serbian",
        "gd": "Scottish Gaelic",
        "sn": "Shona",
        "si": "Sinhala",
        "sk": "Slovak",
        "sl": "Slovene",
        "so": "Somali",
        "st": "Southern Sotho",
        "es": "Spanish",
        "su": "Sundanese",
        "sw": "Swahili",
        "ss": "Swati",
        "sv": "Swedish",
        "ta": "Tamil",
        "te": "Telugu",
        "tg": "Tajik",
        "th": "Thai",
        "ti": "Tigrinya",
        "bo": "Tibetan",
        "tk": "Turkmen",
        "tl": "Tagalog",
        "tn": "Tswana",
        "to": "Tonga",
        "tr": "Turkish",
        "ts": "Tsonga",
        "tt": "Tatar",
        "tw": "Twi",
        "ty": "Tahitian",
        "ug": "Uighur",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "uz": "Uzbek",
        "ve": "Venda",
        "vi": "Vietnamese",
        "vo": "Volapük",
        "wa": "Walloon",
        "cy": "Welsh",
        "wo": "Wolof",
        "fy": "Western Frisian",
        "xh": "Xhosa",
        "yi": "Yiddish",
        "yo": "Yoruba",
        "za": "Zhuang",
        "zu": "Zulu",
    }

    return lang_map[lan]

def read_csv(path_csv):
    csv_reader = pd.read_csv(path_csv, sep="\t", header=0)
    return csv_reader["times"].tolist()


def retrieve_samples(path, dataset):
    sacrebleu = datasets.load_metric("sacrebleu")
    decoders = ["autoregressive", "jacobi", "gs_jacobi", "aw_jacobi"]

    decoder2file = {
        "autoregressive": {"time": [], "trans": []},
        "jacobi": {"time": [], "trans": []},
        "gs_jacobi": {"time": [], "trans": []},
        "aw_jacobi": {"time": [], "trans": []},
    }

    for folder in decoders:
        subf = os.path.join(path, folder)

        for root, dirs, files in os.walk(subf):
            for filename in files:
                if "time" in filename:
                    times = read_csv(os.path.join(root, filename))
                    decoder2file[folder]['time'].extend(times)
                if 'trans' in filename:
                    trans = read_csv(os.path.join(root, filename))
                    decoder2file[folder]['trans'].extend(trans)

    diff_times = np.array([auto-aw for auto, aw in zip(decoder2file['autoregressive']['time'], decoder2file['aw_jacobi']['time'])])
    indices = diff_times.argsort()[::-1]

    for i in indices:
        target = dataset[int(i)][1]
        source = dataset[int(i)][0]
        print("gold", "nd", source)
        print("gold", "nd", target)
        for d in decoders:
            prediction = decoder2file[d]['trans'][i]
            results = sacrebleu.compute(predictions=[prediction], references=[[target]])
            print(d, decoder2file[d]['time'][i], round(results['score'], 3), prediction)

        input()
        continue

def clean_text(text):
    return re.sub("[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~£−€¿]+", " ", text)


def read_wrong_beam_translations(path):
    csv_reader = pd.read_csv(path, sep="\t", header=0)
    idx = csv_reader['i'].tolist()
    bleus = csv_reader['bleu'].tolist()
    beams = csv_reader['beam'].tolist()
    autos = csv_reader['auto'].tolist()
    tgts = csv_reader['tgt'].tolist()

    for x in zip(idx, bleus, beams, autos, tgts):
        print(f"{x[0]} {x[1]}\nBeam: {x[2]}\nAuto: {x[3]}\nTgt: {x[4]}")
        input()
        continue


def write_sentences(path, data):

    with open(path, 'w') as out_file:
        output = csv.writer(out_file, delimiter="\n")
        output.writerow(data)
