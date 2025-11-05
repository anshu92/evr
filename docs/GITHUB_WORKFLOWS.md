# GitHub Workflows Configuration

## Overview

The EVR scanner includes an automated GitHub Actions workflow that runs daily at 6:00 AM Eastern Time, monitors your portfolio, and sends a detailed portfolio report via email.

## Daily Scanner Workflow
**File:** `.github/workflows/daily-scanner.yml`

**Schedule:** Monday-Friday at 6:00 AM Eastern Time (10:00 AM UTC)

**Features:**
- Runs every weekday morning before market open
- Scans all tickers with top 30 recommendations
- Uses intelligent position management (7-day time exits, 20% replacement threshold)
- Caches ticker lists and trained parameters for faster runs
- Sends email with results
- Creates GitHub issue on failure

**Configuration:**
```yaml
Schedule: '0 10 * * 1-5'  # 6:00 AM EDT, 5:00 AM EST
Max Holding Days: 7
Replacement Threshold: 20%
Top Results: 30
```

**Trigger manually:**
```bash
# Via GitHub UI: Actions → Daily EVR Scanner → Run workflow
```

---

## Cache Strategy

The daily workflow implements intelligent caching to improve performance and reduce API calls:

### What's Cached

1. **Python Dependencies** (`~/.cache/uv`, `.venv/`)
   - Key: `{os}-python-{requirements-hash}`
   - Speeds up dependency installation
   - Restored across all workflows

2. **Ticker Data** (`cache/`)
   - Official ticker lists from NASDAQ
   - Delisted ticker tracking
   - Historical price data
   - Reduces API calls to NASDAQ FTP
   - Restored with fallback to most recent cache

3. **Trained Parameters** (`trained_parameters/`)
   - Historical backtesting parameters
   - Setup-specific probabilities
   - Trade statistics
   - Persists learned parameters across runs

### Cache Keys

```yaml
Python: ${{ runner.os }}-python-${{ hashFiles('requirements.txt') }}
Data:   ${{ runner.os }}-evr-cache-${{ github.run_number }}
```

**Restore Strategy:**
- Exact match: Use cached data as-is
- Fallback: Use most recent cache from previous runs
- Miss: Fetch fresh data (slower but complete)

### Cache Benefits

✅ **Faster Runs:** 2-3x faster with warm cache  
✅ **Reduced API Calls:** Preserves ticker lists between runs  
✅ **Cost Savings:** Less data transfer and compute time  
✅ **Reliability:** Fallback to cache if API is slow  
✅ **Consistency:** Same ticker list across daily runs  

---

## Time Zones

### Daily Scanner Schedule

GitHub Actions cron uses UTC time. To run at 6:00 AM Eastern Time:

| Season | Eastern Time | UTC Time | Cron Setting |
|--------|--------------|----------|--------------|
| **EDT** (Mar-Nov) | 6:00 AM | 10:00 AM | `0 10 * * 1-5` ✅ |
| **EST** (Nov-Mar) | 5:00 AM | 10:00 AM | `0 10 * * 1-5` ✅ |

**Configuration:** `0 10 * * 1-5`
- **During EDT (trading season):** Runs at 6:00 AM ET
- **During EST (winter):** Runs at 5:00 AM ET (still before market open at 9:30 AM)

**Rationale:** Using 10:00 AM UTC targets 6:00 AM EDT, which covers most of the trading year (March-November). During winter EST, it runs at 5:00 AM, which is still well before market open.

---

## Email Notifications

### Configuration Required

Set these GitHub Secrets in your repository:
- `EMAIL_USERNAME`: Gmail address for sending
- `EMAIL_PASSWORD`: App-specific password

**To create Gmail app password:**
1. Enable 2-factor authentication on Gmail
2. Go to Google Account → Security
3. App Passwords → Generate new password
4. Use generated password in GitHub secrets

### Email Contents

**Daily/Weekly:**
- Scan summary statistics
- Top recommendations count
- Generated file list
- Workflow run links
- Attached summary file

**Failure Notifications:**
- Error details
- Workflow run information
- Troubleshooting checklist
- Link to GitHub Actions logs

**Recipient:** `anshuman264@gmail.com` (configurable in workflow files)

---

## Failure Handling

All workflows include comprehensive failure handling:

### On Failure

1. **GitHub Issue Created:**
   - Title: "EVR {Daily/Weekly/Manual} Scanner Failed - {date}"
   - Labels: `bug`, `scanner`, `automation`
   - Body: Detailed failure information
   - Links to workflow run

2. **Email Notification Sent:**
   - Alert subject line
   - Failure details
   - Troubleshooting checklist
   - Direct link to logs

3. **Common Issues Checked:**
   - Scanner dependencies
   - Network connectivity
   - Data source availability
   - GitHub Actions configuration

---

## Artifacts

All workflows upload results as GitHub Actions artifacts:

**Retention:** 30 days

**Files Included:**
- CSV files with recommendations
- JSON files with detailed data
- Summary text files
- Generated reports

**Access:**
```
GitHub → Actions → Select Workflow Run → Artifacts section → Download
```

---

## Manual Trigger

You can manually trigger the daily scanner at any time:

```
GitHub → Actions → Daily EVR Scanner → Run workflow → Select branch (main) → Run workflow
```

This will:
- Run the full daily scan immediately
- Monitor and update your portfolio
- Send the portfolio report via email
- Use cached data for faster execution

---

## Monitoring

### Check Workflow Status

**Via GitHub UI:**
```
Repository → Actions → Select workflow → View runs
```

**Check Last Run:**
```
Repository → Actions → Status badge (top of Actions page)
```

### View Logs

**For successful runs:**
```
Actions → Select run → Expand steps → View logs
```

**For failures:**
```
Actions → Select run → Check created issue
Actions → Select run → View email notification
```

### Cache Status

Each run includes a "Check cache status" step that shows:
- Cache directories present
- Ticker list age
- Parameters age
- Cache hit/miss status

---

## Troubleshooting

### Workflow Not Running

**Check:**
1. Repository settings → Actions → Workflows enabled
2. Cron schedule is correct UTC time
3. No GitHub Actions outages
4. Branch has latest workflow files

### Email Not Sending

**Check:**
1. Secrets configured: `EMAIL_USERNAME`, `EMAIL_PASSWORD`
2. Gmail app password valid
3. 2FA enabled on Gmail account
4. Workflow shows email step completed

### Cache Not Working

**Check:**
1. Cache step shows "Cache restored" in logs
2. Cache key matches between runs
3. Cache size within GitHub limits (10 GB)
4. Workflow has cache write permissions

### Scanner Failing

**Check:**
1. Dependencies installed correctly
2. Python version matches (3.11)
3. Network connectivity to data sources
4. NASDAQ FTP accessible
5. yfinance API responding

---

## Best Practices

### 1. Monitor First Week
- Check daily runs complete successfully
- Verify email notifications arrive
- Review generated recommendations
- Confirm cache is working

### 2. Adjust as Needed
- Modify schedule if 6 AM too early/late
- Increase/decrease top signals count
- Adjust position management parameters
- Fine-tune cache retention

### 3. Regular Maintenance
- Review and close automated issues
- Archive old artifacts
- Update dependencies periodically
- Rotate email credentials annually

### 4. Cost Management
- Use cache to minimize API calls
- Limit max_tickers on frequent runs
- Archive artifacts regularly
- Monitor GitHub Actions usage

---

## Customization

### Change Schedule

Edit `.github/workflows/daily-scanner.yml`:
```yaml
schedule:
  - cron: '0 10 * * 1-5'  # Change time here
```

**Cron Examples:**
- `0 9 * * 1-5` - 9:00 AM UTC (5:00 AM EDT)
- `30 10 * * 1-5` - 10:30 AM UTC (6:30 AM EDT)
- `0 14 * * *` - 2:00 PM UTC daily (10:00 AM EDT)

### Change Recipients

Edit email steps in workflow files:
```yaml
to: your-email@example.com  # Change here
```

### Adjust Position Management

Edit scanner command in workflow:
```yaml
--max-holding-days 5 \          # Change from 7
--replacement-threshold 0.15    # Change from 0.20
```

### Add More Tickers

Edit scanner command:
```yaml
--max-tickers 500 \  # Increase from default
```

---

## Performance

### Typical Run Times

| Run Type | First Run (Cold Cache) | Subsequent Runs (Warm Cache) | Speedup |
|----------|----------------------|------------------------------|---------|
| Daily (all tickers) | ~15-20 min | ~8-10 min | 2x faster |

### Cache Impact

**Cold Cache (First Run):**
- Download & install dependencies: 3-5 min
- Fetch ticker lists: 1-2 min
- Download price data: 10-12 min
- **Total:** ~15-20 min

**Warm Cache (Subsequent Runs):**
- Restore dependencies: 30 sec
- Use cached tickers: 5 sec
- Incremental price data: 5-7 min
- **Total:** ~8-10 min

**Savings:** ~50% faster with cache

---

## Security

### Secrets Management
- Never commit credentials to repository
- Use GitHub Secrets for all sensitive data
- Rotate email passwords regularly
- Use app-specific passwords for Gmail

### Repository Access
- Limit who can modify workflows
- Review workflow changes in PRs
- Use branch protection on main
- Enable required reviews

### Cache Security
- Cache is private to repository
- No sensitive data in cache
- Automatic expiration after 7 days unused
- Can be manually cleared if needed

---

## Support

### Questions or Issues

1. Check workflow logs first
2. Review this documentation
3. Check automated GitHub issues
4. Open new issue with details

### Useful Links

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Cron Schedule Helper](https://crontab.guru/)
- [Cache Action Documentation](https://github.com/actions/cache)
- [Email Action Documentation](https://github.com/dawidd6/action-send-mail)

---

**Last Updated:** November 4, 2025  
**Version:** 2.0 - Added intelligent caching and 6AM ET schedule

