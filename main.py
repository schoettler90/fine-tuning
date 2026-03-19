import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = get_device()
    print(f"Active device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    if device == "cuda":
        try:
            from unsloth import FastLanguageModel
            import unsloth
            print(f"Unsloth version: {unsloth.__version__}")
            print("Unsloth (CUDA) ready.")
        except ImportError:
            print("Unsloth not installed — run: uv sync --extra cuda")
    elif device == "mps":
        try:
            from unsloth import FastLanguageModel
            import unsloth
            print(f"Unsloth version: {unsloth.__version__}")
            print("Unsloth (MPS) ready.")
        except ImportError:
            print("Unsloth not installed — run: uv sync --extra mps")
        except NotImplementedError:
            print("Unsloth installed but MPS not supported — falling back to standard PEFT + TRL.")
    else:
        print("No GPU found. CPU-only mode.")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
