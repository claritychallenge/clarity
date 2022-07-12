import json
import logging
import os

import hydra
import speechbrain as sb
import torch
from fastdtw import fastdtw
from hyperpyyaml import load_hyperpyyaml
from omegaconf import DictConfig
from scipy.spatial.distance import cosine
from speechbrain.utils.distributed import run_on_main
from tqdm import tqdm
from transfromer_cpc1_decoder import S2STransformerBeamSearch

logger = logging.getLogger(__name__)


class ASR(sb.core.Brain):
    def generate_feats(self, wavs, wav_lens, tokens_bos):
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

            cnn_out = self.hparams.CNN(feats)
            enc_out, _ = self.hparams.Transformer(
                cnn_out, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
            )
            _, _, dec_out, _ = self.test_search(enc_out.detach(), wav_lens)

        return enc_out.detach().cpu(), dec_out.unsqueeze(0).detach().cpu()

    def init_evaluation(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(max_key=max_key, min_key=min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )
        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

        self.test_search = S2STransformerBeamSearch(
            modules=[
                self.hparams.Transformer,
                self.hparams.seq_lin,
                self.hparams.ctc_lin,
            ],
            bos_index=self.hparams.bos_index,
            eos_index=self.hparams.eos_index,
            blank_index=self.hparams.blank_index,
            min_decode_ratio=self.hparams.min_decode_ratio,
            max_decode_ratio=self.hparams.max_decode_ratio,
            beam_size=self.hparams.test_beam_size,
            ctc_weight=self.hparams.ctc_weight_decode,
            lm_weight=self.hparams.lm_weight,
            lm_modules=self.hparams.lm_model,
            temperature=1,
            temperature_lm=1,
            topk=10,
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


def dtw_similarity(x, y):
    path = fastdtw(
        x.detach().cpu().numpy()[0], y.detach().cpu().numpy()[0], dist=cosine
    )[1]

    x_, y_ = [], []
    for step in range(len(path)):
        x_.append(x[:, path[step][0], :])
        y_.append(y[:, path[step][1], :])
    x_ = torch.stack(x_, dim=1)
    y_ = torch.stack(y_, dim=1)
    return torch.nn.functional.cosine_similarity(x_, y_, dim=-1)


def feat2similarity(
    left_proc_feats, right_proc_feats, left_ref_feats, right_ref_feats, if_dtw=False
):
    if if_dtw:
        ll_sim = dtw_similarity(left_ref_feats, left_proc_feats)
        lr_sim = dtw_similarity(left_ref_feats, right_proc_feats)
        rl_sim = dtw_similarity(right_ref_feats, left_proc_feats)
        rr_sim = dtw_similarity(right_ref_feats, right_proc_feats)
        sim = torch.max(
            torch.stack(
                [
                    torch.mean(ll_sim, dim=-1),
                    torch.mean(lr_sim, dim=-1),
                    torch.mean(rl_sim, dim=-1),
                    torch.mean(rr_sim, dim=-1),
                ],
                dim=-1,
            ),
            dim=-1,
        )[0]
        return sim
    else:
        max_length = torch.max(
            torch.LongTensor(
                [
                    left_proc_feats.shape[1],
                    right_proc_feats.shape[1],
                    left_ref_feats.shape[1],
                    right_ref_feats.shape[1],
                ]
            )
        )
        padded_proc_feats_left = torch.zeros(
            [1, max_length, left_proc_feats.shape[2]], dtype=torch.float32
        )
        padded_proc_feats_right = torch.zeros(
            [1, max_length, right_proc_feats.shape[2]], dtype=torch.float32
        )
        padded_ref_feats_left = torch.zeros(
            [1, max_length, left_ref_feats.shape[2]], dtype=torch.float32
        )
        padded_ref_feats_right = torch.zeros(
            [1, max_length, right_ref_feats.shape[2]], dtype=torch.float32
        )
        padded_proc_feats_left[:, : left_proc_feats.shape[1], :] = left_proc_feats
        padded_proc_feats_right[:, : right_proc_feats.shape[1], :] = right_proc_feats
        padded_ref_feats_left[:, : left_ref_feats.shape[1], :] = left_ref_feats
        padded_ref_feats_right[:, : right_ref_feats.shape[1], :] = right_ref_feats

        ll_sim = torch.nn.functional.cosine_similarity(
            padded_ref_feats_left, padded_proc_feats_left, dim=-1
        )
        lr_sim = torch.nn.functional.cosine_similarity(
            padded_ref_feats_left, padded_proc_feats_right, dim=-1
        )
        rl_sim = torch.nn.functional.cosine_similarity(
            padded_ref_feats_right, padded_proc_feats_left, dim=-1
        )
        rr_sim = torch.nn.functional.cosine_similarity(
            padded_ref_feats_right, padded_proc_feats_right, dim=-1
        )
        sim = torch.stack([ll_sim, lr_sim, rl_sim, rr_sim], dim=-1).max(dim=-1)[0]
        return torch.mean(sim, dim=-1)


def compute_similarity(left_proc_path, wrd, asr_model, bos_index, tokenizer):
    wav_len = torch.tensor([1], dtype=torch.float32)
    tokens_bos = torch.LongTensor([bos_index] + (tokenizer.encode_as_ids(wrd))).view(
        1, -1
    )

    left_ref_path = left_proc_path.replace("msbg", "ref")
    right_proc_path = left_proc_path.replace("left", "right")
    right_ref_path = right_proc_path.replace("msbg", "ref")

    left_proc_wav = sb.dataio.dataio.read_audio(left_proc_path).view(1, -1)
    left_ref_wav = sb.dataio.dataio.read_audio(left_ref_path).view(1, -1)
    right_proc_wav = sb.dataio.dataio.read_audio(right_proc_path).view(1, -1)
    right_ref_wav = sb.dataio.dataio.read_audio(right_ref_path).view(1, -1)

    left_proc_feats = asr_model.generate_feats(left_proc_wav, wav_len, tokens_bos)
    left_ref_feats = asr_model.generate_feats(left_ref_wav, wav_len, tokens_bos)
    right_proc_feats = asr_model.generate_feats(right_proc_wav, wav_len, tokens_bos)
    right_ref_feats = asr_model.generate_feats(right_ref_wav, wav_len, tokens_bos)

    enc_similarity = feat2similarity(
        left_proc_feats[0], right_proc_feats[0], left_ref_feats[0], right_ref_feats[0]
    )
    dec_similarity = feat2similarity(
        left_proc_feats[1],
        right_proc_feats[1],
        left_ref_feats[1],
        right_ref_feats[1],
        if_dtw=True,
    )
    return enc_similarity[0].numpy(), dec_similarity[0].numpy()


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

    # dev set similarity
    dev_enc_similarity = {}
    dev_dec_similarity = {}
    for wav_id, wav_obj in tqdm(left_dev_csv.items()):
        left_proc_path = wav_obj["wav"]
        wrd = wav_obj["wrd"]
        similarity = compute_similarity(
            left_proc_path, wrd, asr_model, bos_index, tokenizer
        )
        dev_enc_similarity[wav_id] = similarity[0].tolist()
        dev_dec_similarity[wav_id] = similarity[1].tolist()

        with open(
            os.path.join(cfg.path.exp_folder, "dev_enc_similarity.json"), "w"
        ) as f:
            json.dump(dev_enc_similarity, f)
        with open(
            os.path.join(cfg.path.exp_folder, "dev_dec_similarity.json"), "w"
        ) as f:
            json.dump(dev_dec_similarity, f)

    # test set similarity
    test_enc_similarity = {}
    test_dec_similarity = {}
    for wav_id, wav_obj in tqdm(left_test_csv.items()):
        left_proc_path = wav_obj["wav"]
        wrd = wav_obj["wrd"]
        similarity = compute_similarity(
            left_proc_path, wrd, asr_model, bos_index, tokenizer
        )
        test_enc_similarity[wav_id] = similarity[0].tolist()
        test_dec_similarity[wav_id] = similarity[1].tolist()

        with open(
            os.path.join(cfg.path.exp_folder, "test_enc_similarity.json"), "w"
        ) as f:
            json.dump(test_enc_similarity, f)
        with open(
            os.path.join(cfg.path.exp_folder, "test_dec_similarity.json"), "w"
        ) as f:
            json.dump(test_dec_similarity, f)


if __name__ == "__main__":
    run()
