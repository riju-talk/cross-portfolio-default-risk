from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipelines.card_pipeline import run_card_pipeline
from pipelines.loan_pipeline import run_loan_pipeline


ARTIFACT_DIR = Path("artifacts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible default-prediction pipelines for credit-card data, "
            "LendingClub loan data, or both."
        )
    )
    parser.add_argument(
        "--problem",
        choices=["card", "loan", "both"],
        default="both",
        help="Which experiment pipeline to run.",
    )
    parser.add_argument(
        "--card-data-path",
        type=Path,
        default=Path("data/default_of_credit_card_clients.csv"),
        help="Path to the credit-card default dataset CSV.",
    )
    parser.add_argument(
        "--loan-data-path",
        type=Path,
        default=Path("data/raw/accepted_2007_to_2018Q4.csv"),
        help="Path to the LendingClub accepted loans CSV.",
    )
    parser.add_argument(
        "--max-loan-rows",
        type=int,
        default=180_000,
        help="Maximum number of terminal-status loan rows to use during training.",
    )
    parser.add_argument(
        "--loan-chunksize",
        type=int,
        default=100_000,
        help="Chunk size used while reading the large accepted loans CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    combined_payload: dict[str, Any] = {
        "executed_problem": args.problem,
        "runs": {},
    }

    if args.problem in {"card", "both"}:
        print("Running card-default pipeline...")
        card_payload = run_card_pipeline(
            data_path=args.card_data_path,
            output_path=ARTIFACT_DIR / "card_metrics.json",
        )
        combined_payload["runs"]["card"] = card_payload
        print(f"Card best model: {card_payload['best_model']['name']}")

    if args.problem in {"loan", "both"}:
        print("Running loan-default pipeline...")
        loan_payload = run_loan_pipeline(
            accepted_path=args.loan_data_path,
            output_path=ARTIFACT_DIR / "loan_metrics.json",
            max_rows=args.max_loan_rows,
            chunksize=args.loan_chunksize,
        )
        combined_payload["runs"]["loan"] = loan_payload
        print(f"Loan best model: {loan_payload['best_model']['name']}")

    combined_path = ARTIFACT_DIR / "metrics.json"
    combined_path.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")

    print("All requested experiments completed.")
    print(f"Combined run summary written to: {combined_path}")


if __name__ == "__main__":
    main()
