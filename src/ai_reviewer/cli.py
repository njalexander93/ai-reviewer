"""Job entry points: prepare, independently review, consolidate, publish and report costs."""

import argparse
import json
import os
from pathlib import Path

from ai_reviewer.contracts import Consolidation, Review, reconcile
from ai_reviewer.github import GitHub, prepare, publish
from ai_reviewer.models import call_model, print_costs


def main() -> None:
    """Run one isolated workflow stage; reject missing inputs without printing secrets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "claude", "openai", "consolidate", "publish", "costs"]
    )
    parser.add_argument("--directory", type=Path, default=Path("results"))
    args = parser.parse_args()
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "costs":
            print_costs(directory)
            return
        context_path = directory / "context.json"
        if args.stage == "prepare":
            # Resolve posting identity from its token, not a hard-coded account name.
            identity = GitHub(os.environ["AI_REVIEWER_TOKEN"]).request("GET", "/user")["login"]
            context = prepare(
                GitHub(os.environ["GH_TOKEN"]),
                os.environ["GITHUB_REPOSITORY"],
                int(os.environ["PR_NUMBER"]),
                identity,
            )
            context_path.write_text(json.dumps(context))
        elif args.stage in ("claude", "openai"):
            call_model(args.stage, context_path.read_text(), directory)
        else:
            reviews = [
                Review.model_validate_json((directory / f"{role}.json").read_text())
                for role in ("claude", "openai")
            ]
            if not all(review.complete for review in reviews):
                raise ValueError("Review incomplete")
            if args.stage == "consolidate":
                payload = {
                    "reviews": [r.model_dump() for r in reviews],
                    "sources": {
                        f"{i}:{j}": f.model_dump()
                        for i, r in enumerate(reviews)
                        for j, f in enumerate(r.findings)
                    },
                }
                call_model("consolidate", json.dumps(payload), directory)
                result = Consolidation.model_validate_json(
                    (directory / "consolidate.json").read_text()
                )
                reconcile(reviews, result)
            else:
                publish(
                    GitHub(os.environ["AI_REVIEWER_TOKEN"]),
                    json.loads(context_path.read_text()),
                    reviews,
                    Consolidation.model_validate_json((directory / "consolidate.json").read_text()),
                )
    except Exception as error:
        # SDK errors can echo request bodies. Log only the exception type.
        print(f"Review stage {args.stage} failed ({type(error).__name__}); no clean verdict.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
