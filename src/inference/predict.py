"""Ad-hoc inference on a single new (headline, image, site) triple, using the
final tuned Hybrid model (exp07). Not part of the training/eval pipeline --
this is for manually trying out new, unseen examples.

Usage:
    python -m src.inference.predict \\
        --headline "כותרת לדוגמה" --image path/to/thumbnail.jpg --site mako

Prints the model's estimated probability that the article would land in the
top 10% of page views for its site, in its publication month.
"""
import argparse

import torch
from PIL import Image

from src.models.hybrid_fallback import HybridClassifier, _EVAL_TRANSFORM
from src.utils import get_device, load_config


def load_model(config_path: str, device):
    cfg = load_config(config_path)
    model = HybridClassifier(cfg).to(device)
    run_dir = f"{cfg['logging']['output_dir']}/{cfg['experiment_name']}"
    state_dict = torch.load(f"{run_dir}/best_model.pt", map_location=device)
    # strict=False: checkpoint only holds the trainable params (frozen ResNet
    # weights are the untouched pretrained ones, loaded fresh above instead).
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def predict(model, headline: str, image_path: str, site: str, device) -> float:
    image = Image.open(image_path).convert("RGB")
    pixel_values = _EVAL_TRANSFORM(image).unsqueeze(0).to(device)

    text_inputs = model.tokenizer(
        [headline], return_tensors="pt", padding=True, truncation=True, max_length=64
    )
    site_idx = torch.tensor([model.site_to_idx[site]], dtype=torch.long, device=device)

    batch = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
        "pixel_values": pixel_values,
        "site_idx": site_idx,
    }
    with torch.no_grad():
        logit = model(batch)
        prob = torch.sigmoid(logit).item()
    return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp07_model2_hybrid_tuned.yaml")
    parser.add_argument("--headline", required=True, help="the article's headline (Hebrew)")
    parser.add_argument("--image", required=True, help="path to a local thumbnail image file")
    parser.add_argument("--site", required=True, choices=["mako", "n12"])
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.config, device)
    prob = predict(model, args.headline, args.image, args.site, device)

    print(f"\nHeadline: {args.headline}")
    print(f"Site: {args.site}")
    print(f"Estimated probability of being a top-10% article: {prob:.1%}")
    print("(for reference: the model was trained so ~10% of real articles are positive,")
    print(" so anything meaningfully above ~10-15% is a relatively strong signal)")


if __name__ == "__main__":
    main()
