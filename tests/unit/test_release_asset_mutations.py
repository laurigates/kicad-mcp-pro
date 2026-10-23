"""Ratchet tests for GitHub Release asset mutations (upstream issue #1179)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.check_github_actions_policy import load_policy
from scripts.check_release_asset_mutations import (
    ReleaseMutation,
    inventory,
    split_shell_commands,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_release_asset_mutations.py"


def _write_workflow(root: Path, body: str, name: str = "test.yml") -> None:
    workflows = root / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    (workflows / name).write_text(body, encoding="utf-8")


def _single_step_workflow(run: str, *, step: str = "Mutate release") -> str:
    indented = "\n".join(f"          {line}" for line in run.splitlines())
    return f"""name: fixture
on: push
permissions:
  contents: read
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - name: {step}
        run: |
{indented}
"""


def _entry(**overrides: object) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "workflow": "test.yml",
        "job": "publish",
        "step": "Mutate release",
        "kind": "upload_asset",
        "clobber": True,
        "issue": "#1179",
        "reason": "Legacy post-publication upload pending draft-first migration.",
    }
    entry.update(overrides)
    return entry


EXPECTED_REPOSITORY_INVENTORY = {
    ("gui-release.yml", "create-release", "Prepare release inventory", "create_release", False),
    (
        "gui-release.yml",
        "create-release",
        "Upload installers and evidence to GitHub Release",
        "upload_asset",
        True,
    ),
    (
        "publish-kicad-pcm.yml",
        "publish",
        "Attach KiCad PCM package and evidence to GitHub Release",
        "upload_asset",
        True,
    ),
    (
        "publish-mcpb.yml",
        "publish",
        "Attach MCPB bundle and evidence to GitHub Release",
        "upload_asset",
        True,
    ),
    ("publish-npm.yml", "publish", "Attach npm evidence to GitHub Release", "upload_asset", True),
    (
        "publish-npm.yml",
        "publish",
        "Attach npm post-publish verification to GitHub Release",
        "upload_asset",
        True,
    ),
    (
        "publish-protocol-schemas.yml",
        "publish",
        "Attach evidence to GitHub Release",
        "upload_asset",
        True,
    ),
    (
        "publish-python.yml",
        "build",
        "Attach Python evidence to GitHub Release",
        "upload_asset",
        True,
    ),
    (
        "publish-release.yml",
        "publish",
        "Publish merged Release Please versions",
        "create_release",
        False,
    ),
}


def test_repository_release_asset_mutations_match_allowlist() -> None:
    entries = inventory(ROOT)

    assert {entry.key for entry in entries} == EXPECTED_REPOSITORY_INVENTORY
    assert all(entry.target_state == "published" for entry in entries)
    policy = load_policy()
    allowlist = policy["release_asset_mutations"]
    assert isinstance(allowlist, list)
    assert len(allowlist) == len(EXPECTED_REPOSITORY_INVENTORY)
    assert all(item["issue"] == "#1179" for item in allowlist)
    assert validate(entries, allowlist) == []


def test_new_clobber_upload_is_rejected(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        _single_step_workflow('gh release upload "$TAG" dist/x.tgz --clobber'),
    )

    entries = inventory(tmp_path)
    errors = validate(entries, [])

    assert entries == [
        ReleaseMutation(
            workflow="test.yml",
            job="publish",
            step="Mutate release",
            kind="upload_asset",
            clobber=True,
            draft=False,
            target_state="published",
            command='gh release upload "$TAG" dist/x.tgz --clobber',
        )
    ]
    assert len(errors) == 1
    message = errors[0]
    assert "test.yml" in message
    assert "publish" in message
    assert "Mutate release" in message
    assert "--clobber" in message
    assert "#1179" in message
    assert "draft" in message


def test_new_upload_without_clobber_onto_non_draft_release_is_rejected(tmp_path: Path) -> None:
    _write_workflow(tmp_path, _single_step_workflow('gh release upload "$TAG" dist/x.tgz'))

    entries = inventory(tmp_path)

    assert [(entry.kind, entry.clobber, entry.target_state) for entry in entries] == [
        ("upload_asset", False, "published")
    ]
    assert len(validate(entries, [])) == 1


def test_upload_into_draft_created_in_same_job_is_allowed(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        """name: fixture
on: push
permissions:
  contents: read
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - name: Create draft release
        run: gh release create "$TAG" --draft --verify-tag --title "$TAG"
      - name: Attach assets to draft
        run: gh release upload "$TAG" dist/*
      - name: Publish draft
        run: gh release edit "$TAG" --draft=false
""",
    )

    entries = inventory(tmp_path)

    assert [(entry.step, entry.kind, entry.target_state) for entry in entries] == [
        ("Create draft release", "create_release", "draft"),
        ("Attach assets to draft", "upload_asset", "draft"),
        ("Publish draft", "edit_release", "draft"),
    ]
    assert validate(entries, []) == []


def test_upload_after_draft_is_published_is_rejected(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        """name: fixture
on: push
permissions:
  contents: read
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - run: gh release create "$TAG" --draft
      - run: gh release edit "$TAG" --draft=false
      - name: Late evidence
        run: gh release upload "$TAG" late.json
""",
    )

    entries = inventory(tmp_path)

    late = [entry for entry in entries if entry.step == "Late evidence"]
    assert [(entry.kind, entry.target_state) for entry in late] == [("upload_asset", "published")]
    assert len(validate(entries, [])) == 1


def test_draft_state_does_not_leak_across_jobs(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        """name: fixture
on: push
permissions:
  contents: read
jobs:
  draft:
    runs-on: ubuntu-latest
    steps:
      - run: gh release create "$TAG" --draft
  attach:
    needs: draft
    runs-on: ubuntu-latest
    steps:
      - name: Attach
        run: gh release upload "$TAG" dist/*
""",
    )

    attach = [entry for entry in inventory(tmp_path) if entry.job == "attach"]

    assert [entry.target_state for entry in attach] == ["published"]


@pytest.mark.parametrize(
    ("run", "kind"),
    [
        ('gh release create "$TAG" --generate-notes', "create_release"),
        ('gh release edit "$TAG" --notes-file notes.md', "edit_release"),
        ('gh release delete-asset "$TAG" old.zip --yes', "delete_asset"),
        ('gh release delete "$TAG" --yes', "delete_release"),
        ('gh api -X POST "repos/$REPO/releases" -f tag_name="$TAG"', "api_write"),
        ('gh api --method PATCH "repos/$REPO/releases/$ID" -F draft=false', "api_write"),
        ('gh api -X DELETE "repos/$REPO/releases/assets/$ASSET"', "api_write"),
        ('gh api "repos/$REPO/releases" -f tag_name="$TAG"', "api_write"),
        (
            'curl -sS -X POST --data-binary @x.zip "https://uploads.github.com/repos/o/r/releases/1/assets?name=x.zip"',
            "upload_asset",
        ),
    ],
)
def test_release_mutation_shell_paths_detected(tmp_path: Path, run: str, kind: str) -> None:
    _write_workflow(tmp_path, _single_step_workflow(run))

    entries = inventory(tmp_path)

    assert [(entry.kind, entry.target_state) for entry in entries] == [(kind, "published")]
    assert len(validate(entries, [])) == 1


@pytest.mark.parametrize(
    ("uses", "with_block", "expected"),
    [
        (
            "googleapis/release-please-action@" + "a" * 40,
            "skip-github-release: false",
            [("create_release", "published")],
        ),
        ("googleapis/release-please-action@" + "a" * 40, "", [("create_release", "published")]),
        ("googleapis/release-please-action@" + "a" * 40, "skip-github-release: true", []),
        (
            "softprops/action-gh-release@" + "b" * 40,
            "files: dist/*",
            [("create_release", "published")],
        ),
        ("softprops/action-gh-release@" + "b" * 40, "draft: true", [("create_release", "draft")]),
        (
            "ncipollo/release-action@" + "c" * 40,
            "allowUpdates: true",
            [("create_release", "published")],
        ),
    ],
)
def test_release_mutation_actions_detected(
    tmp_path: Path, uses: str, with_block: str, expected: list[tuple[str, str]]
) -> None:
    with_yaml = f"\n        with:\n          {with_block}" if with_block else ""
    _write_workflow(
        tmp_path,
        f"""name: fixture
on: push
permissions:
  contents: read
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - name: Release action
        uses: {uses}{with_yaml}
""",
    )

    entries = inventory(tmp_path)

    assert [(entry.kind, entry.target_state) for entry in entries] == expected
    published = [entry for entry in entries if entry.target_state == "published"]
    assert len(validate(entries, [])) == len(published)


@pytest.mark.parametrize(
    "run",
    [
        'gh release view "$TAG" --json isDraft',
        "gh release list --limit 5",
        'gh release download "$TAG" --dir out --clobber',
        'gh release verify-asset "$TAG" dist/x.zip',
        'gh api "repos/$REPO/releases/tags/$TAG"',
        '# gh release upload "$TAG" commented.zip --clobber',
        "echo 'Run gh release upload manually only during recovery'",
    ],
)
def test_read_only_release_commands_ignored(tmp_path: Path, run: str) -> None:
    _write_workflow(tmp_path, _single_step_workflow(run))

    assert inventory(tmp_path) == []


def test_allowlisted_mutation_passes(tmp_path: Path) -> None:
    _write_workflow(tmp_path, _single_step_workflow('gh release upload "$TAG" x --clobber'))

    assert validate(inventory(tmp_path), [_entry()]) == []


def test_clobber_flag_change_requires_allowlist_review(tmp_path: Path) -> None:
    _write_workflow(tmp_path, _single_step_workflow('gh release upload "$TAG" x --clobber'))

    errors = validate(inventory(tmp_path), [_entry(clobber=False)])

    assert any("not allowlisted" in error for error in errors)
    assert any("stale" in error for error in errors)


def test_stale_allowlist_entry_is_rejected(tmp_path: Path) -> None:
    _write_workflow(tmp_path, _single_step_workflow("echo nothing to release"))

    errors = validate(inventory(tmp_path), [_entry()])

    assert len(errors) == 1
    assert "stale" in errors[0]
    assert "Mutate release" in errors[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"issue": ""},
        {"issue": "1179"},
        {"issue": "#abc"},
        {"reason": ""},
        {"reason": "   "},
        {"kind": "rename_release"},
        {"clobber": "yes"},
        {"step": ""},
    ],
)
def test_allowlist_entries_require_issue_reference_and_reason(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    _write_workflow(tmp_path, _single_step_workflow('gh release upload "$TAG" x --clobber'))

    errors = validate(inventory(tmp_path), [_entry(**overrides)])

    assert any("release_asset_mutations[0]" in error for error in errors)


def test_allowlist_must_be_a_list() -> None:
    errors = validate([], {"workflow": "x"})

    assert errors == ["actions-policy.json: release_asset_mutations must be a list"]


def test_multiline_run_and_continuations_parsed(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        _single_step_workflow(
            """for attempt in $(seq 1 18); do
  if gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
    gh release upload "$RELEASE_TAG" \\
      "$upload_dir"/* \\
      --clobber
    exit 0
  fi
done"""
        ),
    )

    entries = inventory(tmp_path)

    assert [(entry.kind, entry.clobber) for entry in entries] == [("upload_asset", True)]


def test_split_shell_commands_handles_separators_and_comments() -> None:
    script = (
        "set -e\na && gh release upload t x; echo done # trailing\n  # only a comment\nb || c | d"
    )

    assert split_shell_commands(script) == [
        "set -e",
        "a",
        "gh release upload t x",
        "echo done # trailing",
        "b",
        "c",
        "d",
    ]


def test_unnamed_step_uses_id_then_index(tmp_path: Path) -> None:
    _write_workflow(
        tmp_path,
        """name: fixture
on: push
permissions:
  contents: read
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - id: upload-one
        run: gh release upload "$TAG" a --clobber
      - run: gh release upload "$TAG" b --clobber
""",
    )

    assert [entry.step for entry in inventory(tmp_path)] == ["upload-one", "#2"]


def test_cli_reports_inventory_and_fails_on_unlisted_mutation(tmp_path: Path) -> None:
    _write_workflow(tmp_path, _single_step_workflow('gh release upload "$TAG" x --clobber'))
    policy = tmp_path / ".github" / "actions-policy.json"
    policy.write_text(json.dumps({"release_asset_mutations": []}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--policy", str(policy)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "not allowlisted" in result.stdout

    policy.write_text(json.dumps({"release_asset_mutations": [_entry()]}), encoding="utf-8")
    ok = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--policy", str(policy)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "1 allowlisted post-publication mutation" in ok.stdout


def test_cli_passes_on_repository() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stdout + result.stderr
