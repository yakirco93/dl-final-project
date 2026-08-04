"""Compare two (headline, image) candidates for the same article/site --
e.g. two real A/B-test variants -- and see which one the final model (exp07)
prefers. Meant for manually validating the model against real, known A/B
test outcomes: does the model's higher-scored variant match the variant
that actually won in production?

Usage:
    python -m src.inference.compare_ab \\
        --site mako \\
        --headline-a "כותרת אפשרות א" --image-a path/to/image_a.jpg \\
        --headline-b "כותרת אפשרות ב" --image-b path/to/image_b.jpg
"""
import argparse

from src.inference.predict import load_model, predict
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp07_model2_hybrid_tuned.yaml")
    parser.add_argument("--site", required=True, choices=["mako", "n12"])
    parser.add_argument("--headline-a", required=True)
    parser.add_argument("--image-a", required=True)
    parser.add_argument("--headline-b", required=True)
    parser.add_argument("--image-b", required=True)
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.config, device)

    prob_a = predict(model, args.headline_a, args.image_a, args.site, device)
    prob_b = predict(model, args.headline_b, args.image_b, args.site, device)

    print(f"\nOption A: {args.headline_a}")
    print(f"  probability: {prob_a:.1%}")
    print(f"\nOption B: {args.headline_b}")
    print(f"  probability: {prob_b:.1%}")

    winner = "A" if prob_a > prob_b else ("B" if prob_b > prob_a else "tie")
    print(f"\nModel prefers: {winner}")
    print("Compare this pick against the real A/B test outcome to check agreement.")


if __name__ == "__main__":
    main()
