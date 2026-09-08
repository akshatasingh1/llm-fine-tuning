"""
One-off: push the trained LoRA adapter to the HF Hub so the Space can load it.

    hf auth login                  # paste a WRITE token from hf.co/settings/tokens
    python demo/upload_adapter.py --adapter_dir results/sql_lora --repo <HF_USERNAME>/qwen2.5-0.5b-sql-lora
"""
import argparse

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", default="results/sql_lora")
    ap.add_argument("--repo", required=True, help="e.g. yourname/qwen2.5-0.5b-sql-lora")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=args.adapter_dir, repo_id=args.repo, repo_type="model",
        allow_patterns=["adapter_*", "*.json", "*.jinja", "tokenizer*", "*.txt", "*.md"],
        ignore_patterns=["checkpoint-*/*", "*optimizer*", "*.pt", "*.pth", "*.bin"],
    )
    print(f"Uploaded {args.adapter_dir} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
