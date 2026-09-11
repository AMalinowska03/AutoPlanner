import json
import os
import re
from pathlib import Path

# Ścieżka do katalogu
base_dir = Path("PPOGenerated")
output_file = Path("finished_user_ids.json")

# 1. Znalezienie folderów i wyciągnięcie ID
pattern = re.compile(r"^finetune_u(\d+)$")
finished_user_ids = []

if base_dir.exists() and base_dir.is_dir():
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                finished_user_ids.append(int(match.group(1)))

# Posortowanie i wypisanie znalezionych ID
finished_user_ids.sort()
print(f"Znalezione ID ({len(finished_user_ids)}): \n{finished_user_ids}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(finished_user_ids, f, indent=4)