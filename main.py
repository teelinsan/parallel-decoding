import hydra
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBartForConditionalGeneration,
)

from src.bench import MTBenchmarker
from src.dataset.flores_dataset import Flores
from src.dataset.ittb_dataset import Ittb
from src.dataset.iwslt_dataset import Iwslt
from src.dataset.wmt_dataset import Wmt
from src.ipi.decoders.autoregressive import AutoregressiveDecoder
from src.ipi.decoders.beam_search import BeamSearchDecoder
from src.ipi.decoders.gs_jacobi import GSJacobiDecoder
from src.ipi.decoders.hybrid_jacobi import HybridJacobiDecoder
from src.ipi.decoders.jacobi import JacobiDecoder
from src.ipi.initializer import Initializer
from src.ipi.decoders.mt_decoding import MTDecoder
from src.utils.beam_search import BeamSearcher
from src.utils.utils import retrieve_samples


def load_tokenizer(cfg):
    # MBart
    mapping_dict = {
        "ar": "ar_AR",
        "cs": "cs_CZ",
        "de": "de_DE",
        "en": "en_XX",
        "es": "es_XX",
        "et": "et_EE",
        "fi": "fi_FI",
        "fr": "fr_XX",
        "gu": "gu_IN",
        "hi": "hi_IN",
        "it": "it_IT",
        "ja": "ja_XX",
        "kk": "kk_KZ",
        "ko": "ko_KR",
        "lt": "lt_LT",
        "lv": "lv_LV",
        "my": "my_MM",
        "ne": "ne_NP",
        "nl": "nl_XX",
        "ro": "ro_RO",
        "ru": "ru_RU",
        "si": "si_LK",
        "tr": "tr_TR",
        "vi": "vi_VN",
        "zh": "zh_CN",
        "af": "af_ZA",
        "az": "az_AZ",
        "bn": "bn_IN",
        "fa": "fa_IR",
        "he": "he_IL",
        "hr": "hr_HR",
        "id": "id_ID",
        "ka": "ka_GE",
        "km": "km_KH",
        "mk": "mk_MK",
        "ml": "ml_IN",
        "mn": "mn_MN",
        "mr": "mr_IN",
        "pl": "pl_PL",
        "ps": "ps_AF",
        "pt": "pt_XX",
        "sv": "sv_SE",
        "sw": "sw_KE",
        "ta": "ta_IN",
        "te": "te_IN",
        "th": "th_TH",
        "tl": "tl_XX",
        "uk": "uk_UA",
        "ur": "ur_PK",
        "xh": "xh_ZA",
        "gl": "gl_ES",
        "sl": "sl_SI",
    }

    if "mbart" in cfg.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            src_lang=mapping_dict[cfg.src_lang],
            tgt_lang=mapping_dict[cfg.tgt_lang],
            use_fast=False,
        )
    else:
        print(cfg.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    return tokenizer


def load_model(cfg):
    if "mbart" in cfg.model_name:
        model = MBartForConditionalGeneration.from_pretrained(cfg.model_name).to(
            cfg.device
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name).to(cfg.device)

    return model


def load_dataset(tokenizer, cfg):
    # Wmt-xx-xx-xx
    if cfg.name == "wmt":
        split = cfg.split
        if cfg.subset.use_subset:
            split = f"{cfg.split}[{cfg.subset.start}:{cfg.subset.end + 1}]"

        dataset = Wmt(
            version=cfg.version,
            src_lan=cfg.src_lang,
            tgt_lan=cfg.tgt_lang,
            hugginface_tokenizer=tokenizer,
            split=split,
        )
    # Iwsltxx-xx-xx
    elif cfg.name == "iwslt":
        dataset = Iwslt(
            data_dir=cfg.data_dir,
            version=str(cfg.version),
            src_lan=cfg.src_lang,
            tgt_lan=cfg.tgt_lang,
            hugginface_tokenizer=tokenizer,
            split=cfg.split,
        )
    elif cfg.name == "ittb":
        dataset = Ittb(
            src_lan=cfg.src_lang,
            tgt_lan=cfg.tgt_lang,
            hugginface_tokenizer=tokenizer,
            split=cfg.split,
        )
    elif cfg.name == "flores":
        dataset = Flores(
            src_lan=cfg.src_lang,
            tgt_lan=cfg.tgt_lang,
            hugginface_tokenizer=tokenizer,
            split=cfg.split,
        )
    else:
        raise ValueError(f"{cfg.dataset.name} is not yet implemented")

    return dataset


def load_initializer(tokenizer, cfg):
    if cfg.use_initializer:
        initializer = Initializer(
            src_len=cfg.src_lang,
            tgt_len=cfg.tgt_lang,
            hugginface_tokenizer=tokenizer,
            use_init=cfg.use_init,
            device=cfg.device,
        )

    else:
        initializer = None

    return initializer


def compute_beam_search(cfg, model, dataset):
    initializer = load_initializer(dataset.tokenizer, cfg.initializer)

    decoder = MTDecoder(
        tokenizer=dataset.tokenizer,
        model=model,
        use_cache=cfg.decoder.use_cache,
        gs_jaco_blocks=cfg.decoder.gs_jaco_blocks,
        device=cfg.device,
        initializer=initializer
    )

    beam_searcher = BeamSearcher(
        model=model,
        dataset=dataset,
        initializer=initializer,
        decoder=decoder,
        batch_size=cfg.beam_search.batch_size,
        num_beams=cfg.beam_search.num_beams,
        device=cfg.beam_search.device,
        no_repeat_ngram_size=2,
        early_stopping=True,
        result_dir=cfg.beam_search.result_dir,
    )

    beam_searcher.compute_beam_search(cfg)


def load_decoders(cfg, tokenizer, model, initializer):
    decoders = []
    for decoder in cfg.decoder.decoders:
        if decoder == "autoregressive":
            dec = AutoregressiveDecoder(
                tokenizer=tokenizer,
                model=model,
                initializer=initializer,
                use_cache=cfg.decoder.use_cache,
                device=cfg.decoder.device
            )
        elif decoder == "jacobi":
            dec = JacobiDecoder(
                tokenizer=tokenizer,
                model=model,
                initializer=initializer,
                use_cache=cfg.decoder.use_cache,
                device=cfg.decoder.device
            )
        elif decoder == "gs_jacobi":
            dec = GSJacobiDecoder(
                tokenizer=tokenizer,
                model=model,
                initializer=initializer,
                gs_jaco_blocks=cfg.decoder.gs_jaco_blocks,
                init_mode="",
                use_cache=cfg.decoder.use_cache,
                device=cfg.decoder.device
            )
        elif decoder == "h_jacobi":
            dec = HybridJacobiDecoder(
                tokenizer=tokenizer,
                model=model,
                initializer=initializer,
                init_mode="fixed",
                gs_jaco_blocks=cfg.decoder.gs_jaco_blocks,
                use_cache=cfg.decoder.use_cache,
                device=cfg.decoder.device
            )
        elif decoder == "beam_search":
            dec = BeamSearchDecoder(
                tokenizer=tokenizer,
                model=model,
                initializer=initializer,
                num_beams=cfg.beam_search.num_beams,
                early_stopping=True,
            )
        else:
            raise ValueError(f"{decoder} decoder have not been implemented yet.")

        decoders.append(dec)

    return decoders


def compute_benchmark(cfg, tokenizer, dataset, model):
    initializer = load_initializer(tokenizer, cfg.initializer)
    decoders = load_decoders(cfg, tokenizer, model, initializer)

    benchmarker = MTBenchmarker(
        dataset=dataset,
        decoders=decoders,
        src_lang=cfg.model.src_lang,
        tgt_lang=cfg.model.tgt_lang,
        result_dir=cfg.bench.result_dir,
        device=cfg.bench.device,
        debug=True,
    )
    benchmarker.compute_total_time()


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg):
    tokenizer = load_tokenizer(cfg.model)

    model = load_model(cfg.model)

    dataset = load_dataset(tokenizer, cfg.dataset)

    if "benchmark" in cfg.task:
        compute_benchmark(cfg, tokenizer, dataset, model)
    elif "beam_search" in cfg.task:
        compute_beam_search(cfg, model, dataset)
    elif "sample" in cfg.task:
        retrieve_samples(cfg.sample.path, dataset)
    else:
        raise ValueError(f"{cfg.task} is not yet implemented")


if __name__ == "__main__":
    main()
