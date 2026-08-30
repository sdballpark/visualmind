from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
from huggingface_hub import model_info
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor


MODEL_ID = "google/siglip2-base-patch16-224"
CATALOG = Path("data/metadata/image_catalog.csv")
MODEL_CONFIG = Path("configs/models.yaml")


def get_feature_tensor(value):
    if isinstance(value, torch.Tensor):
        return value

    if hasattr(value, "pooler_output"):
        return value.pooler_output

    raise TypeError(
        f"Unexpected get_image_features() result: {type(value)}"
    )


def choose_test_image() -> Path:
    with CATALOG.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)

        for row in rows:
            source = Path(row["source_path"])

            if (
                source.exists()
                and source.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ):
                return source

    raise RuntimeError("No suitable test image found in catalog.")


def main() -> None:
    print()
    print("=" * 72)
    print("VISUALMIND - SIGLIP2 GPU SMOKE TEST")
    print("=" * 72)

    hf_home = os.environ.get("HF_HOME")

    print(f"HF_HOME:        {hf_home}")
    print(f"Model:          {MODEL_ID}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA runtime:   {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    print(f"GPU:            {torch.cuda.get_device_name(0)}")

    print()
    print("Resolving exact Hugging Face model revision...")

    info = model_info(MODEL_ID)
    revision = info.sha

    print(f"Revision:       {revision}")

    MODEL_CONFIG.parent.mkdir(parents=True, exist_ok=True)

    MODEL_CONFIG.write_text(
        f"""models:
  image_embedding:
    provider: huggingface
    repo_id: {MODEL_ID}
    revision: {revision}
    purpose: semantic image-text retrieval
    precision: float16
""",
        encoding="utf-8",
    )

    print(f"Model manifest: {MODEL_CONFIG.resolve()}")

    test_image = choose_test_image()

    print()
    print(f"Test image:     {test_image.name}")
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=revision,
    )

    print("Loading SigLIP2 model...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=revision,
        torch_dtype=torch.float16,
    )

    model = model.to(device)
    model.eval()

    print("Model device:   ", next(model.parameters()).device)
    print(
        "GPU memory after load:",
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
    )

    print()
    print("Opening local test image...")

    with Image.open(test_image) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    print("Generating image embedding on GPU...")

    with torch.inference_mode():
        result = model.get_image_features(**inputs)

    embedding = get_feature_tensor(result)

    embedding = embedding / embedding.norm(
        p=2,
        dim=-1,
        keepdim=True,
    )

    embedding = embedding.float().cpu()

    print()
    print("-" * 72)
    print("SMOKE TEST RESULT")
    print("-" * 72)

    print(f"Embedding shape: {tuple(embedding.shape)}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(
        f"L2 norm:         "
        f"{embedding.norm(p=2, dim=-1).item():.6f}"
    )

    print(
        "GPU peak memory: ",
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
    )

    print()
    print("First 8 embedding values:")
    print(embedding[0, :8].tolist())

    print()
    print("=" * 72)
    print("SIGLIP2 GPU SMOKE TEST PASSED")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
