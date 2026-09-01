"""Generate VLM captions for every image in the catalog.

Writes data/metadata/captions.csv. Safe to re-run: images already
captioned are skipped, so an interrupted run resumes where it stopped.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

CATALOG = Path("data/metadata/image_catalog.csv")
MODEL_CONFIG = Path("configs/models.yaml")
OUTPUT = Path("data/metadata/captions.csv")

PROMPT = (
    "Describe this photo in one or two sentences. "
    "Name the people, animals, objects, and setting. "
    "State only what is visible; omit mood and interpretation."
)

FIELDNAMES = ["source_path", "filename", "caption", "model", "revision"]


def load_model_config():
    config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    entry = config["models"]["vision_language"]
    return entry["repo_id"], entry["revision"]


def load_done() -> set[str]:
    if not OUTPUT.exists():
        return set()
    with OUTPUT.open("r", encoding="utf-8", newline="") as handle:
        return {row["source_path"] for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Caption at most N images (0 = all).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-caption everything, discarding the existing file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=320,
        help=(
            "Ceiling, not a target: greedy decoding stops at EOS, so "
            "this only ever truncates. At 100 it truncated a quarter of "
            "this corpus mid-sentence."
        ),
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1024 * 28 * 28,
        help=(
            "Cap on vision tokens per image. The corpus contains 45 MP "
            "photos; without this, a single image can exhaust VRAM."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    repo, revision = load_model_config()

    with CATALOG.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    done = set() if args.force else load_done()
    todo = [r for r in rows if r["source_path"] not in done]

    if args.limit:
        todo = todo[:args.limit]

    print()
    print("=" * 76)
    print("VISUALMIND - BUILD VLM CAPTIONS")
    print("=" * 76)
    print(f"Catalog images:  {len(rows)}")
    print(f"Already done:    {len(done)}")
    print(f"To caption:      {len(todo)}")
    print(f"Model:           {repo}")
    print(f"Revision:        {revision}")
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Max pixels:      {args.max_pixels:,}")

    if not todo:
        print("\nNothing to do.")
        return 0

    print("\nLoading model from shared HF cache...")
    proc = AutoProcessor.from_pretrained(
        repo, revision=revision, max_pixels=args.max_pixels
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        repo, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = args.force or not OUTPUT.exists()

    started = time.time()
    failures = 0

    with OUTPUT.open(
        "w" if args.force else "a", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)

        if write_header:
            writer.writeheader()

        for number, row in enumerate(todo, start=1):
            path = Path(row["source_path"])

            try:
                image = Image.open(path).convert("RGB")
            except Exception as error:
                print(f"  SKIP {path.name}: {error}")
                failures += 1
                continue

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ]}]

            inputs = proc.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to("cuda")

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            caption = proc.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            writer.writerow({
                "source_path": row["source_path"],
                "filename": row["filename"],
                "caption": caption,
                "model": repo,
                "revision": revision,
            })
            handle.flush()

            if number % 25 == 0 or number == len(todo):
                rate = (time.time() - started) / number
                remaining = (len(todo) - number) * rate
                print(f"  {number}/{len(todo)}  "
                      f"{rate:.1f}s/image  "
                      f"~{remaining/60:.0f} min left")

    elapsed = time.time() - started

    print()
    print("-" * 76)
    print("CAPTION SUMMARY")
    print("-" * 76)
    print(f"Captioned:       {len(todo) - failures}")
    print(f"Failed:          {failures}")
    print(f"Elapsed:         {elapsed/60:.1f} min")
    print(f"GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"\nCaptions: {OUTPUT.resolve()}")
    print()
    print("=" * 76)
    print("CAPTIONING COMPLETE")
    print("=" * 76)

    return 0


if __name__ == "__main__":
    sys.exit(main())
