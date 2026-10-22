"""Inventory and ratchet GitHub Release mutations made by workflows (upstream #1179).

GitHub immutable releases lock a release's assets once it is published. Any
workflow step that creates a non-draft release, or adds, replaces or deletes
assets on a release that is already published, is incompatible with that
setting. This checker statically inventories such steps in
``.github/workflows/*.y*ml`` and fails when a step mutates a published release
without an explicit allowlist entry in ``.github/actions-policy.json``
(``release_asset_mutations``). Allowlist entries that no longer match a step
also fail, so the migration to draft -> attach -> publish can only ratchet
forward.

The check is a deterministic text/YAML ratchet, not a proof: it recognises
``gh release``, ``gh api`` and ``curl`` release writes, and the known release
Actions. Indirect invocations (``eval``, variables holding ``gh``) are out of
scope; see ``docs/development/release-asset-immutability.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_KEY = "release_asset_mutations"
ISSUE = "#1179"
DOC_PATH = "docs/development/release-asset-immutability.md"

KINDS = frozenset(
    {
        "create_release",
        "upload_asset",
        "edit_release",
        "delete_release",
        "delete_asset",
        "api_write",
    }
)
_GH_RELEASE_KINDS = {
    "create": "create_release",
    "upload": "upload_asset",
    "edit": "edit_release",
    "delete": "delete_release",
    "delete-asset": "delete_asset",
}
_RELEASE_ACTIONS = {
    "googleapis/release-please-action",
    "softprops/action-gh-release",
    "ncipollo/release-action",
}
_SHELL_PREFIX_TOKENS = {
    "!",
    "(",
    "{",
    "if",
    "then",
    "else",
    "elif",
    "do",
    "while",
    "until",
    "time",
    "exec",
    "command",
}
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEPARATORS = re.compile(r"\n|&&|\|\||;|\||\$\(|`")
_WRITE_METHODS = {"POST", "PATCH", "PUT", "DELETE"}
_GH_API_BODY_FLAGS = {"-f", "-F", "--field", "--raw-field", "--input"}
_CURL_BODY_FLAGS = {"-d", "--data", "--data-binary", "--data-raw", "-F", "--form", "-T"}
_ISSUE_REF = re.compile(r"^#[1-9][0-9]*$")


@dataclass(frozen=True)
class ReleaseMutation:
    """One workflow step that writes to a GitHub Release."""

    workflow: str
    job: str
    step: str
    kind: str
    clobber: bool
    draft: bool
    target_state: str
    command: str

    @property
    def key(self) -> tuple[str, str, str, str, bool]:
        return (self.workflow, self.job, self.step, self.kind, self.clobber)

    @property
    def location(self) -> str:
        return f"{self.workflow}:{self.job}:{self.step}"


def split_shell_commands(script: str) -> list[str]:
    """Split a ``run`` script into simple command segments.

    Backslash continuations are joined, full-line comments are dropped and the
    script is split on newlines, ``;``, ``&&``, ``||``, pipes and command
    substitution openers. Quoting is not honoured when splitting; this only
    affects which segment a token lands in, not whether a command starts with
    ``gh``/``curl``.
    """

    joined = re.sub(r"\\\r?\n", " ", script)
    segments: list[str] = []
    for line in joined.splitlines():
        if line.strip().startswith("#"):
            continue
        for part in _SEPARATORS.split(line):
            text = " ".join(part.split())
            if text:
                segments.append(text)
    return segments


def _tokens(segment: str) -> list[str]:
    try:
        tokens = shlex.split(segment, comments=False)
    except ValueError:
        tokens = segment.split()
    while tokens and (tokens[0] in _SHELL_PREFIX_TOKENS or _ENV_ASSIGNMENT.match(tokens[0])):
        tokens = tokens[1:]
    return tokens


def _flag_value(tokens: list[str], short: str | None, long: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == long or (short is not None and token == short):
            if index + 1 < len(tokens):
                return tokens[index + 1]
        if token.startswith(f"{long}="):
            return token.split("=", 1)[1]
        if short is not None and token.startswith(short) and len(token) > len(short):
            if not token.startswith("--"):
                return token[len(short) :]
    return None


def _draft_flag(tokens: list[str]) -> bool | None:
    """Return True for ``--draft``, False for ``--draft=false``, None when absent."""

    for arg in tokens:
        if arg in {"--draft", "-d", "--draft=true"}:
            return True
        if arg == "--draft=false":
            return False
    return None


@dataclass(frozen=True)
class _Match:
    kind: str
    clobber: bool
    draft: bool | None
    command: str


def _classify_segment(segment: str) -> _Match | None:
    tokens = _tokens(segment)
    if len(tokens) < 2:
        return None
    program = tokens[0].rsplit("/", 1)[-1]
    if program == "gh" and tokens[1] == "release" and len(tokens) >= 3:
        kind = _GH_RELEASE_KINDS.get(tokens[2])
        if kind is None:
            return None
        rest = tokens[3:]
        return _Match(kind, "--clobber" in rest, _draft_flag(rest), segment)
    if program == "gh" and tokens[1] == "api":
        rest = tokens[2:]
        if not any("releases" in token for token in rest):
            return None
        method = _flag_value(rest, "-X", "--method")
        if method is None:
            method = "POST" if any(token in _GH_API_BODY_FLAGS for token in rest) else "GET"
        if method.upper() in _WRITE_METHODS:
            return _Match("api_write", False, None, segment)
        return None
    if program == "curl":
        rest = tokens[1:]
        if any("uploads.github.com" in token for token in rest):
            return _Match("upload_asset", False, None, segment)
        if any("/releases" in token for token in rest):
            method = _flag_value(rest, "-X", "--request")
            if method is None and any(token in _CURL_BODY_FLAGS for token in rest):
                method = "POST"
            if method is not None and method.upper() in _WRITE_METHODS:
                return _Match("api_write", False, None, segment)
    return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _classify_action(step: Mapping[str, object]) -> _Match | None:
    uses = step.get("uses")
    if not isinstance(uses, str):
        return None
    action = uses.split("@", 1)[0].lower()
    if action not in _RELEASE_ACTIONS:
        return None
    raw_with = step.get("with")
    inputs: Mapping[str, object] = raw_with if isinstance(raw_with, dict) else {}
    if action == "googleapis/release-please-action":
        if _truthy(inputs.get("skip-github-release", False)):
            return None
        return _Match("create_release", False, None, uses)
    draft = _truthy(inputs.get("draft", False))
    return _Match("create_release", False, draft, uses)


def _step_label(step: Mapping[str, object], index: int) -> str:
    for field in ("name", "id"):
        value = step.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"#{index}"


def _step_matches(step: Mapping[str, object]) -> list[_Match]:
    matches: list[_Match] = []
    action_match = _classify_action(step)
    if action_match is not None:
        matches.append(action_match)
    run = step.get("run")
    if isinstance(run, str):
        for segment in split_shell_commands(run):
            match = _classify_segment(segment)
            if match is not None:
                matches.append(match)
    return matches


def _job_inventory(workflow: str, job: str, steps: Iterable[object]) -> list[ReleaseMutation]:
    entries: list[ReleaseMutation] = []
    seen: set[tuple[str, str, str, str, bool]] = set()
    draft_open = False
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        label = _step_label(step, index)
        for match in _step_matches(step):
            draft = bool(match.draft)
            if match.kind == "create_release":
                target = "draft" if draft else "published"
                draft_open = draft
            elif match.kind == "edit_release" and draft_open:
                target = "draft"
                if match.draft is False:
                    draft_open = False
            else:
                target = "draft" if draft_open else "published"
            entry = ReleaseMutation(
                workflow=workflow,
                job=job,
                step=label,
                kind=match.kind,
                clobber=match.clobber,
                draft=draft,
                target_state=target,
                command=match.command,
            )
            if entry.key in seen:
                continue
            seen.add(entry.key)
            entries.append(entry)
    return entries


def inventory(root: Path) -> list[ReleaseMutation]:
    """Return every GitHub Release mutation found in the repository workflows."""

    entries: list[ReleaseMutation] = []
    for path in sorted((root / ".github" / "workflows").glob("*.y*ml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            # Parse failures are reported by check_github_actions_policy.py.
            continue
        if not isinstance(data, dict):
            continue
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            steps = job.get("steps")
            if isinstance(steps, list):
                entries.extend(_job_inventory(path.name, str(job_name), steps))
    return entries


def _allowlist_key(
    index: int, item: object, errors: list[str]
) -> tuple[str, str, str, str, bool] | None:
    prefix = f"actions-policy.json: {POLICY_KEY}[{index}]"
    if not isinstance(item, dict):
        errors.append(f"{prefix} must be an object")
        return None
    problems: list[str] = []
    for field in ("workflow", "job", "step"):
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{field} must be a non-empty string")
    if item.get("kind") not in KINDS:
        problems.append(f"kind must be one of {sorted(KINDS)}")
    if not isinstance(item.get("clobber"), bool):
        problems.append("clobber must be a boolean")
    issue = item.get("issue")
    if not isinstance(issue, str) or not _ISSUE_REF.fullmatch(issue):
        problems.append(f"issue must reference a tracking issue such as {ISSUE!r}")
    reason = item.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append("reason must explain why the mutation is tolerated")
    if problems:
        errors.append(f"{prefix}: " + "; ".join(problems))
        return None
    return (
        str(item["workflow"]),
        str(item["job"]),
        str(item["step"]),
        str(item["kind"]),
        bool(item["clobber"]),
    )


def validate(entries: Iterable[ReleaseMutation], allowlist: object) -> list[str]:
    """Return ratchet errors for post-publication mutations versus the allowlist."""

    if not isinstance(allowlist, list):
        return [f"actions-policy.json: {POLICY_KEY} must be a list"]

    errors: list[str] = []
    allowed: dict[tuple[str, str, str, str, bool], int] = {}
    for index, item in enumerate(allowlist):
        key = _allowlist_key(index, item, errors)
        if key is None:
            continue
        if key in allowed:
            errors.append(
                f"actions-policy.json: {POLICY_KEY}[{index}] duplicates entry {allowed[key]}"
            )
            continue
        allowed[key] = index

    matched: set[tuple[str, str, str, str, bool]] = set()
    for entry in entries:
        if entry.target_state != "published":
            continue
        if entry.key in allowed:
            matched.add(entry.key)
            continue
        clobber = " with --clobber" if entry.clobber else ""
        errors.append(
            f"{entry.location}: {entry.kind}{clobber} mutates a published (non-draft) GitHub "
            f"Release and is not allowlisted in actions-policy.json {POLICY_KEY}. Immutable "
            f"releases ({ISSUE}) need draft -> attach -> publish: create the release as a "
            "draft, attach every asset without --clobber, then publish it; see "
            f"{DOC_PATH}. Command: {entry.command}"
        )

    for key, index in sorted(allowed.items(), key=lambda pair: pair[1]):
        if key in matched:
            continue
        workflow, job, step, kind, uses_clobber = key
        clobber_text = str(uses_clobber).lower()
        errors.append(
            f"actions-policy.json: {POLICY_KEY}[{index}] is stale: no published-release "
            f"mutation matches {workflow}:{job}:{step} ({kind}, clobber={clobber_text}). "
            f"Remove the entry once the step is migrated ({ISSUE} ratchet)."
        )
    return errors


def _load_allowlist(policy_path: Path, root: Path) -> object:
    resolved_root = root.resolve(strict=True)
    candidate = policy_path if policy_path.is_absolute() else resolved_root / policy_path
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SystemExit("ERROR: policy must be a file inside --root") from exc
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("ERROR: policy root must be an object")
    return data.get(POLICY_KEY, [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(".github") / "actions-policy.json",
        help="Policy JSON inside --root (default: .github/actions-policy.json).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full inventory as JSON and exit 0."
    )
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve(strict=True)
    entries = inventory(root)
    if args.json:
        print(
            json.dumps(
                [entry.__dict__ for entry in entries],
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    errors = validate(entries, _load_allowlist(args.policy, root))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    published = sum(1 for entry in entries if entry.target_state == "published")
    workflows = len({entry.workflow for entry in entries})
    print(
        f"Release asset mutation ratchet passed: {published} allowlisted post-publication "
        f"mutation(s) across {workflows} workflow(s) ({ISSUE})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
