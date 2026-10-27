"""Job entry points: prepare, independently review, consolidate, publish and report costs."""

import argparse
import os
from pathlib import Path

from ai_reviewer.consolidation import consolidate_chunks
from ai_reviewer.context import prepare
from ai_reviewer.contracts import (
    FinalConsolidation,
    ReviewContext,
)
from ai_reviewer.diagnostics import report_failure
from ai_reviewer.github import GitHub, publish
from ai_reviewer.models import ConfigurationError, format_costs, print_costs
from ai_reviewer.work import load_reviews, review_chunks


def main() -> None:
    """Run the stage selected by command-line arguments.

    Reads the selected stage's environment credentials and JSON artifacts, then
    writes artifacts, calls external APIs or prints costs as that stage requires.
    Stage failures record redacted metadata without exposing input data.

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
                repository_directory=directory / "repository",
                # Empty workflow inputs mean the caller supplied no expectation.
                expected_head=os.environ.get("EXPECTED_HEAD_SHA") or None,
                expected_base=os.environ.get("EXPECTED_BASE_SHA") or None,
            )
            # Persist the collected packet for the enabled independent reviewers.
            context_path.write_text(context.model_dump_json())
        else:
            context = ReviewContext.model_validate_json(context_path.read_text())
            if args.stage == "review":
                review_chunks(context, directory)
            elif args.stage == "consolidate":
                consolidate_chunks(context, directory)
            else:
                publish(
                    GitHub(os.environ["AI_REVIEWER_TOKEN"]),
                    context,
                    load_reviews(context, directory),
                    FinalConsolidation.model_validate_json(
                        (directory / "consolidate.json").read_text()
                    ),
                    cost_report=format_costs(directory),
                )
    except Exception as error:
        # Keep stage failures diagnosable without printing arbitrary exception text.
        if isinstance(error, ConfigurationError):
            print(str(error))
        report_failure(error, directory / f"{args.stage}.error.json", stage=args.stage)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
