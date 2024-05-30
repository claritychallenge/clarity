from local.tasnet import ConvTasNetStereo
import torch
import yaml
import os

conf_path = os.path.join("/home/gerardoroadabike/Downloads/", "conf.yml")
with open(conf_path) as f:
    conf = yaml.safe_load(f)

model = ConvTasNetStereo(
    **conf["convtasnet"],
    samplerate=conf["data"]["sample_rate"],
)

saved = torch.load("/home/gerardoroadabike/Downloads/causal.pth", map_location="cpu")
model.load_state_dict(saved["state_dict"])

model.save_pretrained("model_noncausal")

model.push_to_hub(
    "cadenzachallenge/ConvTasNet_Lyrics_NonCausal",
    token="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",
)
