"""
Smoke test: confirms kagglehub can pull a file from each of the two
project datasets. Run with the venv's Python:

    .venv\\Scripts\\python.exe test_data_pull.py

Requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY env vars) — see https://github.com/Kaggle/kagglehub#authenticate
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASETS = [
    {
        "name": "VCT 2025 All Events (International + Regional)",
        "slug": "piyush86kumar/valorant-vct-2025-all-events",
        "file_path": "VCT 2025 Americas Stage 1_csvs/matches.csv",
    },
    {
        "name": "VCT 2021-2026 Data",
        "slug": "ryanluong1/valorant-champion-tour-2021-2023-data",
        "file_path": "vct_2025/agents/teams_picked_agents.csv",
    },
]


def main():
    for dataset in DATASETS:
        print(f"\n=== {dataset['name']} ===")
        print(f"slug: {dataset['slug']}")
        print(f"file: {dataset['file_path']}")

        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            dataset["slug"],
            dataset["file_path"],
        )

        print(f"shape: {df.shape}")
        print(df.head())


if __name__ == "__main__":
    main()
