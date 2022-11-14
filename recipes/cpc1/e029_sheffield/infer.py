import copy
import json
import logging
import os

import hydra
import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from omegaconf import DictConfig
from speechbrain.utils.distributed import run_on_main
from tqdm import tqdm
from transformer_cpc1_ensemble_decoder import S2STransformerBeamSearch

logger = logging.getLogger(__name__)


class ASR(sb.core.Brain):
    def compute_uncertainty(self, wavs, wav_lens, tokens_bos):
        """Forward computations from the waveform batches to the output probabilities."""
        # batch = batch.to(self.device)
        wavs, wav_lens, tokens_bos = (
            wavs.to(self.device),
            wav_lens.to(self.device),
            tokens_bos.to(self.device),
        )
        with torch.no_grad():
            feats = self.hparams.compute_features(wavs)
            current_epoch = self.hparams.epoch_counter.current
            feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

            cnn_out = []
            for j in range(self.hparams.n_ensembles):
                cnn_out.append(self.asr_ensemble[j][0](feats))
            _, _, prob_outputs = self.test_search(
                cnn_out, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
            )

        return prob_outputs

    def init_ensembles(self, n_ensemble):
        ensembles = []
        for j in range(n_ensemble):
            ensembles.append(copy.deepcopy(self.hparams.model))
        return ensembles

    def init_evaluation(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        self.asr_ensemble = self.init_ensembles(self.hparams.n_ensembles)
        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        test_search_modules = []
        for j in range(self.hparams.n_ensembles):
            ckpt = sb.utils.checkpoints.average_checkpoints(
                [ckpts[j]], recoverable_name="model", device=self.device
            )
            self.asr_ensemble[j].load_state_dict(ckpt, strict=True)
            self.asr_ensemble[j].eval()
            test_search_modules.extend(self.asr_ensemble[j][1:])

        self.test_search = S2STransformerBeamSearch(
            modules=test_search_modules,
            n_ensembles=self.hparams.n_ensembles,
            bos_index=self.hparams.bos_index,
            eos_index=self.hparams.eos_index,
            blank_index=self.hparams.blank_index,
            min_decode_ratio=self.hparams.min_decode_ratio,
            max_decode_ratio=self.hparams.max_decode_ratio,
            beam_size=self.hparams.test_beam_size,
            ctc_weight=self.hparams.ctc_weight_decode,
            lm_weight=self.hparams.lm_weight,
            lm_modules=self.hparams.lm_model,
            temperature=self.hparams.temperature,
            temperature_lm=1,
            topk=self.hparams.topk,
            using_eos_threshold=False,
            length_normalization=True,
        )


def init_asr(asr_config):
    hparams_file, run_opts, overrides = sb.parse_arguments([asr_config])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    tokenizer = hparams["tokenizer"]
    bos_index = hparams["bos_index"]

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.init_evaluation()

    return asr_brain, tokenizer, bos_index


def compute_uncertainty(left_proc_path, asr_model, bos_index, tokenizer):
    wav_len = torch.tensor([1], dtype=torch.float32)
    tokens_bos = torch.LongTensor([bos_index]).view(1, -1)

    right_proc_path = left_proc_path.replace("left", "right")
    left_proc_wav = sb.dataio.dataio.read_audio(left_proc_path).view(1, -1)
    right_proc_wav = sb.dataio.dataio.read_audio(right_proc_path).view(1, -1)

    left_uncertainty = asr_model.compute_uncertainty(left_proc_wav, wav_len, tokens_bos)
    right_uncertainty = asr_model.compute_uncertainty(
        right_proc_wav, wav_len, tokens_bos
    )
    conf = max(
        left_uncertainty[0]["confidence"].detach().cpu().numpy(),
        right_uncertainty[0]["confidence"].detach().cpu().numpy(),
    )
    neg_ent = -min(
        left_uncertainty[0]["entropy"].detach().cpu().numpy(),
        right_uncertainty[0]["entropy"].detach().cpu().numpy(),
    )
    return conf, neg_ent


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig) -> None:
    if cfg.cpc1_track == "open":
        track = "_indep"
    elif cfg.cpc1_track == "closed":
        track = ""
    else:
        logger.error("cpc1_track has to be closed or open")

    asr_model, tokenizer, bos_index = init_asr(cfg.asr_config)

    left_dev_csv = sb.dataio.dataio.load_data_csv(
        os.path.join(cfg.path.exp_folder, "cpc1_asr_data" + track, "left_dev_msbg.csv")
    )  # using left ear csvfile for data loading
    left_test_csv = sb.dataio.dataio.load_data_csv(
        os.path.join(cfg.path.exp_folder, "cpc1_asr_data" + track, "left_test_msbg.csv")
    )  # using left ear csvfile for data loading

    dev_conf = {}
    dev_negent = {}
    # dev set uncertainty
    for wav_id, wav_obj in tqdm(left_dev_csv.items()):
        left_proc_path = wav_obj["wav"]
        uncertainty = compute_uncertainty(
            left_proc_path, asr_model, bos_index, tokenizer
        )
        dev_conf[wav_id] = uncertainty[0].tolist()
        dev_negent[wav_id] = uncertainty[1].tolist()

        with open(os.path.join(cfg.path.exp_folder, "dev_conf.json"), "w") as f:
            json.dump(dev_conf, f)
        with open(os.path.join(cfg.path.exp_folder, "dev_negent.json"), "w") as f:
            json.dump(dev_negent, f)

    # test set similarity
    test_conf = {}
    test_negent = {}
    for wav_id, wav_obj in tqdm(left_test_csv.items()):
        left_proc_path = wav_obj["wav"]
        uncertainty = compute_uncertainty(
            left_proc_path, asr_model, bos_index, tokenizer
        )
        test_conf[wav_id] = uncertainty[0].tolist()
        test_negent[wav_id] = uncertainty[1].tolist()

        with open(os.path.join(cfg.path.exp_folder, "test_conf.json"), "w") as f:
            json.dump(test_conf, f)
        with open(os.path.join(cfg.path.exp_folder, "test_negent.json"), "w") as f:
            json.dump(test_negent, f)


if __name__ == "__main__":
    run()
