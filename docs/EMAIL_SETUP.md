# Email Setup for Daily Reports

## Overview

The EVR scanner can automatically send daily portfolio reports via email. This requires setting up GitHub Secrets with your email credentials.

This repository also includes a **Daily Stock Screener (CAD) + Portfolio Weights** workflow that emails:
- `reports/daily_report.txt`
- `reports/portfolio_weights.csv`

If you enable ML scoring (default in the daily screener workflow), make sure you run the training workflow at least once so the model exists.

## Quick Setup (5 minutes)

### Step 1: Enable 2-Factor Authentication on Gmail

1. Go to your [Google Account](https://myaccount.google.com/)
2. Click **Security** in the left sidebar
3. Under "How you sign in to Google", click **2-Step Verification**
4. Follow the prompts to enable 2FA
5. ✅ You'll need this before creating an app password

---

### Step 2: Generate App-Specific Password

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Scroll down to "How you sign in to Google"
3. Click **App Passwords** (only visible if 2FA is enabled)
4. You may need to sign in again
5. In the "Select app" dropdown, choose **Mail**
6. In the "Select device" dropdown, choose **Other** and type "EVR Scanner"
7. Click **Generate**
8. 📋 **Copy the 16-character password** (looks like: `xxxx xxxx xxxx xxxx`)
9. ⚠️ Save this password - you won't see it again!

---

### Step 3: Add Secrets to GitHub Repository

1. Go to your GitHub repository: `https://github.com/anshu92/evr`
2. Click **Settings** (top right)
3. In left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**

**First Secret:**
- Name: `EMAIL_USERNAME`
- Value: Your Gmail address (e.g., `anshuman264@gmail.com`)
- Click **Add secret**

**Second Secret:**
- Click **New repository secret** again
- Name: `EMAIL_PASSWORD`
- Value: The 16-character app password from Step 2 (remove spaces)
- Click **Add secret**

**Optional Secret (Recipient):**
- Name: `EMAIL_TO`
- Value: Your recipient email address
- If not set, the workflows default to sending to `EMAIL_USERNAME`

---

### Step 4: Verify Setup

1. Go to **Actions** tab in your repository
2. Select **Daily EVR Scanner** workflow
3. Click **Run workflow** → **Run workflow**
4. Wait for workflow to complete (~8-10 minutes)
5. Check your email inbox for: **"⚡ EVR Daily Action Plan + Portfolio Report"**

---

## What You'll Receive

### Email Schedule
📧 **Every weekday at 6:00 AM ET** (10:00 AM UTC)

### Email Contents

**Subject:** ⚡ EVR Daily Action Plan + Portfolio Report

**Attachment:** `portfolio_report.txt` with:

1. **⚡ RECOMMENDED ACTIONS** (Top of report)
   - Specific BUY instructions (shares, entry, stop, target)
   - Specific SELL instructions (shares, reason)
   - Priority levels (🔴 HIGH, 🟡 MEDIUM, 🟢 LOW)

2. **Portfolio Summary**
   - Total capital, available capital, P&L, returns

3. **Open Positions**
   - Entry price, stop loss, target, days held
   - Expected return, risk dollars

4. **Recently Closed Positions**
   - Exit price, P&L, return %
   - Exit reason (TIME_EXIT, STOPPED_OUT, TARGET_HIT, REPLACED)

5. **Performance Statistics**
   - Total closed, wins, losses, win rate

---

## Example Email Report

```
======================================================================
EVR DAILY PORTFOLIO REPORT & ACTION PLAN
======================================================================

⚡ RECOMMENDED ACTIONS
======================================================================

1. 🔴 SELL 20 shares of ORN
   Reason: Time exit (held 7 days)
   Current Entry: $10.88

2. 🔴 BUY 15 shares of NOC
   Entry: $575.23 | Stop: $549.79 | Target: $626.11
   Reason: EVR Score: 0.744, 8 signals
   Cost: $8,628.45

3. 🟡 BUY 12 shares of UNFI
   Entry: $36.83 | Stop: $34.44 | Target: $41.61
   Reason: EVR Score: 0.728, 7 signals
   Cost: $441.96

PORTFOLIO SUMMARY
----------------------------------------------------------------------
Total Capital:        $1,000.00
Available Capital:    $135.65 (13.6%)
Total P&L:            +$45.23
Total Return:         +4.52%
Open Positions:       5

[... rest of report ...]
```

---

## Troubleshooting

### ❌ Email Not Arriving

**Check 1: Secrets Configured**
```
GitHub → Settings → Secrets and variables → Actions
Verify EMAIL_USERNAME and EMAIL_PASSWORD exist
```

**Check 2: Workflow Logs**
```
GitHub → Actions → Daily EVR Scanner → Latest run
Look for: "✅ Email secrets configured. Email will be sent."
```

**Check 3: Gmail Spam Folder**
- Check spam/junk folder
- Mark as "Not Spam" if found there

**Check 4: App Password Valid**
- Regenerate app password if needed
- Update EMAIL_PASSWORD secret with new password

---

### ❌ Workflow Shows "Email secrets not configured"

This means the secrets aren't set up. Follow Steps 1-3 above.

---

### ❌ "Input required and not supplied: from"

Old error - fixed in latest version. Update your repository:
```bash
git pull origin main
```

---

### ❌ App Passwords Option Not Showing

You need 2-Factor Authentication enabled first:
1. Go to Google Account → Security
2. Enable 2-Step Verification
3. Wait 24 hours (sometimes required)
4. App Passwords option will appear

---

## Security Notes

### ✅ Safe
- ✅ App-specific passwords are safe for GitHub Secrets
- ✅ Limited to mail access only
- ✅ Can be revoked anytime
- ✅ Different from your main Google password
- ✅ GitHub Secrets are encrypted

### ⚠️ Best Practices
- Don't share the app password
- Don't commit passwords to code
- Revoke and regenerate if compromised
- Use a dedicated email for automated reports (optional)

---

## Alternative Email Providers

### Using Other Email Services

The workflow currently uses Gmail but can be adapted:

**For Outlook/Hotmail:**
```yaml
server_address: smtp-mail.outlook.com
server_port: 587
```

**For Yahoo:**
```yaml
server_address: smtp.mail.yahoo.com
server_port: 587
```

**For Custom SMTP:**
```yaml
server_address: your.smtp.server.com
server_port: 587  # or 465 for SSL
```

Update `.github/workflows/daily-scanner.yml` with your provider's settings.

---

## Disable Email Notifications

If you don't want email notifications:

1. Simply don't set up the EMAIL_USERNAME and EMAIL_PASSWORD secrets
2. Workflow will run successfully but skip email step
3. You'll see: "⚠️ Email secrets not configured. Skipping email notification."
4. Portfolio report and scan results still available in GitHub Actions artifacts

---

## FAQ

**Q: Is my email password secure?**  
A: Yes! GitHub Secrets are encrypted and never exposed in logs. Plus, you're using an app-specific password, not your main Google password.

**Q: Can I use a different email address?**  
A: Yes! The report is sent TO `anshuman264@gmail.com`, but you can change this in the workflow file.

**Q: How do I change the recipient?**  
A: Edit `.github/workflows/daily-scanner.yml`, find `to: anshuman264@gmail.com`, and change it to your desired email address.

**Q: Can I get emails at a different time?**  
A: Yes! The workflow runs at 6:00 AM ET. Edit the cron schedule in the workflow file.

**Q: What if I want multiple recipients?**  
A: Edit the workflow and use comma-separated emails: `to: email1@gmail.com,email2@gmail.com`

**Q: Can I disable the attachment?**  
A: Edit the workflow and remove the `attachments: portfolio_report.txt` line.

---

## Support

If you're still having issues:

1. Check [GitHub Actions logs](https://github.com/anshu92/evr/actions)
2. Review [Workflow Documentation](GITHUB_WORKFLOWS.md)
3. Open an issue on the repository

---

**Ready to get started? Follow Steps 1-4 above! 📧**

