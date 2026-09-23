#!/usr/bin/env python3
"""Publish Taut's checkpoint mirrors to Hugging Face.

Taut is a fork of Laya. The weights are upstream's, Apache-2.0, and are republished
**unmodified** under this project's own account so the package has a stable home for them
and does not depend on another account's repo names staying put.

    python scripts/mirror_checkpoints.py --dry-run     # show the plan, touch nothing
    python scripts/mirror_checkpoints.py               # ask, then publish

Needs a Hugging Face token with write access to the target account:

    huggingface-cli login          # or export HF_TOKEN=hf_...

Nothing is retrained, quantised or altered. Each mirror carries a model card naming the
upstream source, the licence and the commit it was taken from, because a mirror that does
not say what it mirrors is indistinguishable from a claim of authorship.
"""
import argparse
import os
import sys

UPSTREAM_ORG = "convaiinnovations"
DEFAULT_TARGET_ORG = os.environ.get("TAUT_MODEL_ORG", "thekarteek")

# (upstream repo, taut repo suffix, what it is)
CHECKPOINTS = [
    ("laya", "taut", "English, ModernBERT-large, 421M, 512 tokens"),
    ("laya-multilingual", "taut-multilingual", "100+ languages, mmBERT-base, 322M, 1024 tokens"),
    ("laya-typed-decisions", "taut-typed-decisions",
     "typed-decisions workflows, ModernBERT-large, 421M, 1024 tokens"),
]

CARD = """---
license: apache-2.0
library_name: taut
tags:
  - decision-model
  - conformal-prediction
  - risk-control
  - text-classification
---

# {target}

An **unmodified mirror** of [`{upstream}`]({upstream_url}), republished so
[Taut](https://github.com/thedataengineer/taut) has a stable home for the weights it loads.

{description}

## Provenance

| | |
|---|---|
| Upstream repository | [`{upstream}`]({upstream_url}) |
| Upstream project | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), Convai Innovations |
| Upstream revision mirrored | `{revision}` |
| Licence | Apache-2.0, unchanged |
| Modifications | **none** — no retraining, quantisation, pruning or config edits |

The model architecture, the RLCD training recipe and these weights are upstream's work. If
you use them, cite the upstream project.

## What Taut adds

Taut is a fork that leaves the model alone and builds on top of the score: certified,
distribution-free risk control over the decisions (`taut.conformal`) and label-free
detection of a gate that has stopped applying (`taut.drift`).

```python
from taut import Agent, ConformalGate

gate = ConformalGate.calibrate(results, labels, alpha=0.02, delta=0.05)
# "at most 2% of decisions are auto-handled incorrectly, with 95% confidence"
```

See the [repository](https://github.com/thedataengineer/taut) for the guarantee and the
evidence it holds.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-org", default=DEFAULT_TARGET_ORG,
                        help="Hugging Face account to publish into (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without contacting the hub for writes")
    parser.add_argument("--private", action="store_true",
                        help="create the mirrors private rather than public")
    parser.add_argument("--only", action="append", metavar="NAME",
                        help="mirror only this checkpoint (repeatable); default is all three")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt (for non-interactive use)")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
        return 2

    wanted = [c for c in CHECKPOINTS if not args.only or c[1] in args.only or c[0] in args.only]
    if not wanted:
        print("no checkpoint matched --only %s" % args.only, file=sys.stderr)
        return 2

    api = HfApi()
    token = os.environ.get("HF_TOKEN")

    print("Mirroring %d checkpoint(s) into %s/  (%s)\n"
          % (len(wanted), args.target_org, "private" if args.private else "public"))
    plan = []
    for upstream_name, target_name, description in wanted:
        upstream = "%s/%s" % (UPSTREAM_ORG, upstream_name)
        target = "%s/%s" % (args.target_org, target_name)
        try:
            info = api.model_info(upstream)
            revision = info.sha
            size = sum(getattr(f, "size", 0) or 0 for f in (info.siblings or []))
        except Exception as e:                            # noqa: BLE001
            print("  ! %-40s could not read upstream: %s" % (upstream, e), file=sys.stderr)
            return 2
        plan.append((upstream, target, description, revision))
        print("  %-42s -> %s" % (upstream, target))
        print("     %s" % description)
        print("     upstream revision %s%s" % (revision[:12],
                                               ("  ~%.0f MB" % (size / 1e6)) if size else ""))
    print()

    if args.dry_run:
        print("Dry run: nothing was created or uploaded.")
        return 0

    if not token and not os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
        print("No Hugging Face credentials found. Run `huggingface-cli login` or set HF_TOKEN.",
              file=sys.stderr)
        return 2

    if not args.yes:
        # Publishing is outward-facing and hard to undo quietly, so it is confirmed.
        reply = input("Publish these %d mirror(s) to %s? [y/N] " % (len(plan), args.target_org))
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted; nothing was created or uploaded.")
            return 1

    for upstream, target, description, revision in plan:
        print("\n== %s" % target)
        print("   creating repo")
        api.create_repo(target, repo_type="model", private=args.private,
                        exist_ok=True, token=token)
        print("   downloading %s @ %s" % (upstream, revision[:12]))
        local = snapshot_download(upstream, revision=revision, token=token)
        card = CARD.format(target=target, upstream=upstream, description=description,
                           revision=revision,
                           upstream_url="https://huggingface.co/" + upstream)
        card_path = os.path.join(local, "README.md")
        # The snapshot lives in the shared hub cache; write the card beside it without
        # clobbering upstream's own file in that cache.
        import tempfile, shutil
        staging = tempfile.mkdtemp(prefix="taut-mirror-")
        shutil.copytree(local, staging, dirs_exist_ok=True, symlinks=False)
        with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(card)
        print("   uploading")
        api.upload_folder(repo_id=target, folder_path=staging, repo_type="model",
                          token=token,
                          commit_message="Mirror %s @ %s (unmodified, Apache-2.0)"
                                         % (upstream, revision[:12]))
        shutil.rmtree(staging, ignore_errors=True)
        print("   done -> https://huggingface.co/%s" % target)

    print("\nAll mirrors published. `import taut` will now find them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
