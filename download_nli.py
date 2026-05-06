import os
from huggingface_hub import snapshot_download

model_name = "cross-encoder/nli-deberta-v3-base"
local_dir = os.path.join("Model", "nli-deberta-v3-base")

print(f"Downloading {model_name} to {local_dir}...")
snapshot_download(repo_id=model_name, local_dir=local_dir)
print("Download complete!")
