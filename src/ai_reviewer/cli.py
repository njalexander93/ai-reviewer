"""Job entry points: prepare, independently review, consolidate, publish and report costs."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_reviewer.chunking import plan
from ai_reviewer.context import prepare
from ai_reviewer.contracts import (
    Consolidation,
    FinalConsolidation,
    Review,
    ReviewContext,
    reconcile,
    source_findings,
)
from ai_reviewer.github import GitHub, publish, validate_anchors
from ai_reviewer.models import call_model, print_costs
from ai_reviewer.work import consolidate_chunks, load_reviews, review_chunks


def main() -> None:
    """Run the stage selected by command-line arguments.

    Reads the selected stage's environment credentials and JSON artifacts, then
    writes artifacts, calls external APIs or prints costs as that stage requires.
    Stage failures print only their exception type to avoid exposing input data.

    Raises:
        SystemExit: Argument parsing rejects the invocation, or a stage fails
            with exit code 1. Help requests exit successfully through argparse.
        OSError: The output directory cannot be created before stage execution.
    """
    # Select one workflow stage and its shared artifact directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["prepare", "review", "consolidate", "publish", "costs"],
    )
    parser.add_argument("--directory", type=Path, default=Path("results"))
    args = parser.parse_args()
    # Prepare storage before entering the sanitized stage-error boundary.
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    # Keep provider error bodies out of Actions logs for every dispatched stage.
    try:
        # Cost reporting needs no context packet or provider credentials.
        if args.stage == "costs":
            print_costs(directory)
            return
        # All review stages share the packet produced by preparation.
        context_path = directory / "context.json"
        if args.stage == "prepare":
            # Resolve posting identity from its token, not a hard-coded account name.
            identity = GitHub(os.environ["AI_REVIEWER_TOKEN"]).request("GET", "/user")["login"]
            context = prepare(
                GitHub(os.environ["GH_TOKEN"]),
                os.environ["GITHUB_REPOSITORY"],
                int(os.environ["PR_NUMBER"]),
                identity,
                linear_token=os.environ.get("LINEAR_API_KEY"),
            )
            # Persist the collected packet for both independent reviewers.
            context_path.write_text(context.model_dump_json())
        # Start both paid calls before waiting; a failure still waits for the other usage record.
        elif args.stage == "review":
            context = ReviewContext.model_validate_json(context_path.read_text())
            if plan(context).whole is None:
                review_chunks(context, directory)
                return
            payload = context.model_dump_json()
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(call_model, role, payload, directory)
                    for role in ("claude", "openai")
                ]
                # Propagate failures only after the executor has allowed both calls to finish.
                for future in futures:
                    future.result()
        else:
            context = ReviewContext.model_validate_json(context_path.read_text())
            if plan(context).whole is None:
                if args.stage == "consolidate":
                    consolidate_chunks(context, directory)
                else:
                    reviews, _ = load_reviews(context, directory)
                    publish(
                        GitHub(os.environ["AI_REVIEWER_TOKEN"]),
                        context,
                        reviews,
                        FinalConsolidation.model_validate_json(
                            (directory / "consolidate.json").read_text()
                        ),
                        directory=directory,
                    )
                return
            # Downstream stages require both validated independent reports.
            reviews = [
                Review.model_validate_json((directory / f"{role}.json").read_text())
                for role in ("claude", "openai")
            ]
            # Use the shared source contract before spending on consolidation.
            if args.stage == "consolidate":
                sources = source_findings(reviews)
                validate_anchors(
                    ReviewContext.model_validate_json(context_path.read_text()), reviews
                )
                consolidation_payload = {
                    "reviews": [{"summary": r.summary} for r in reviews],
                    "sources": {key: f.model_dump() for key, f in sources.items()},
                }
                call_model("consolidate", json.dumps(consolidation_payload), directory)
                # Reject the stage if the written consolidation fails source preservation.
                result = Consolidation.model_validate_json(
                    (directory / "consolidate.json").read_text()
                )
                reconcile(reviews, result)
            else:
                # Publication performs its own identity, revision and finding checks.
                publish(
                    GitHub(os.environ["AI_REVIEWER_TOKEN"]),
                    ReviewContext.model_validate_json(context_path.read_text()),
                    reviews,
                    Consolidation.model_validate_json((directory / "consolidate.json").read_text()),
                )
    except Exception as error:
        # SDK errors can echo request bodies. Log only the exception type.
        print(f"Review stage {args.stage} failed ({type(error).__name__}); no clean verdict.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
