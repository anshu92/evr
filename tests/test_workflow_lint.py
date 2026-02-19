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


def test_daily_workflow_writes_health_summary_counters():
    text = Path(".github/workflows/daily-stock-screener.yml").read_text(encoding="utf-8")
    assert "id: health_counters" in text
    assert "Write workflow summary" in text
    assert "GITHUB_STEP_SUMMARY" in text
    assert "cache/reward_log.json" in text
    assert "cache/action_reward_log.json" in text
    assert "screener_portfolio_state.json" in text
    assert "reports/trade_actions.json" in text
    assert "steps.health_counters.outputs.state_age_hours" in text


def test_cache_prune_workflow_exists_and_targets_daily_cache_keys():
    workflow_path = Path(".github/workflows/prune-actions-caches.yml")
    assert workflow_path.exists()
    text = workflow_path.read_text(encoding="utf-8")
    assert "schedule:" in text
    assert "workflow_dispatch:" in text
    assert "actions: write" in text
    assert "actions/github-script@v7" in text
    assert "-daily-screener-data-" in text
    assert "DELETE /repos/{owner}/{repo}/actions/caches/{cache_id}" in text
    assert "keep_latest" in text
    assert "max_age_days" in text
    assert "GITHUB_STEP_SUMMARY" in text


def test_daily_coverage_workflow_exists_and_checks_session_windows():
    workflow_path = Path(".github/workflows/verify-daily-session-coverage.yml")
    assert workflow_path.exists()
    text = workflow_path.read_text(encoding="utf-8")
    assert "schedule:" in text
    assert "cron: '15 22 * * 1-5'" in text
    assert "workflow_dispatch:" in text
    assert "daily-stock-screener.yml" in text
    assert "PRE_MARKET" in text
    assert "MID_DAY" in text
    assert "PRE_CLOSE" in text
    assert "listWorkflowRuns" in text
    assert "Missing session windows" in text
    assert "issues: write" in text
    assert "Daily Screener Coverage Gap -" in text
    assert "GITHUB_STEP_SUMMARY" in text


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
