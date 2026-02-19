from pathlib import Path
import re


def test_daily_workflow_uses_restore_save_cache_pattern():
    text = Path(".github/workflows/daily-stock-screener.yml").read_text(encoding="utf-8")
    assert "key: ${{ runner.os }}-daily-screener-data-${{ github.run_id }}" not in text
    assert "uses: actions/cache/restore@v4" in text
    assert "uses: actions/cache/save@v4" in text
    assert "id: cache_data_restore" in text
    assert "steps.cache_data_restore.outputs.cache-hit" in text
    assert (
        "key: ${{ runner.os }}-daily-screener-data-${{ hashFiles('requirements-actions.txt', '.github/workflows/daily-stock-screener.yml') }}-${{ github.run_id }}"
        in text
    )


def test_daily_workflow_model_artifact_selection_hardened():
    text = Path(".github/workflows/daily-stock-screener.yml").read_text(encoding="utf-8")
    assert "branch: defaultBranch" in text
    assert "per_page: 50" in text


def test_daily_workflow_has_concurrency_guard():
    text = Path(".github/workflows/daily-stock-screener.yml").read_text(encoding="utf-8")
    assert "concurrency:" in text
    assert "group: daily-stock-screener" in text
    assert "cancel-in-progress: false" in text


def test_train_workflow_cache_keys_are_not_run_id_based():
    text = Path(".github/workflows/train-stock-screener-model.yml").read_text(encoding="utf-8")
    assert "key: ${{ runner.os }}-screener-model-${{ github.run_id }}" not in text


def test_docs_match_weekly_training_schedule():
    readme = Path("README.md").read_text(encoding="utf-8").lower()
    workflows_doc = Path("docs/GITHUB_WORKFLOWS.md").read_text(encoding="utf-8").lower()
    assert "weekly" in readme
    assert "weekly" in workflows_doc
    assert "every ~3 days" not in workflows_doc


def test_actions_minutes_estimate_within_private_free_tier():
    """Budget guardrail for private-repo free tier (2,000 Linux minutes/month)."""
    daily_text = Path(".github/workflows/daily-stock-screener.yml").read_text(encoding="utf-8")
    train_text = Path(".github/workflows/train-stock-screener-model.yml").read_text(encoding="utf-8")

    m = re.search(r'MAX_DAILY_RUNTIME_MINUTES:\s*"?(?P<v>\d+)"?', daily_text)
    assert m is not None, "MAX_DAILY_RUNTIME_MINUTES must be pinned in workflow env"
    daily_budget_min = int(m.group("v"))

    # 3 weekday runs/day * ~22 weekdays/month.
    monthly_daily_runs = 3 * 22
    monthly_daily_minutes = monthly_daily_runs * daily_budget_min

    # Weekly training at timeout upper bound.
    m_train = re.search(r"timeout-minutes:\s*(?P<v>\d+)", train_text)
    assert m_train is not None
    monthly_train_minutes = 4 * int(m_train.group("v"))

    total_estimated = monthly_daily_minutes + monthly_train_minutes
    assert total_estimated <= 2000
