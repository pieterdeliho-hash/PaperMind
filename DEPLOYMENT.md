\# PaperMind Deployment Guide



\## Live Production Instance

\*\*URL\*\*: https://papermind-ai-assistant.streamlit.app



\---



\## Prerequisites



\### Required Accounts

1\. \*\*GitHub Account\*\* (for code hosting)

2\. \*\*Streamlit Cloud Account\*\* (for deployment)

3\. \*\*OpenAI API Key\*\* (for GPT-3.5-turbo)



\### Local Requirements

\- Python 3.11+

\- Git

\- 8GB+ RAM (for embedding models)



\---



\## Local Development Setup



\### 1. Clone Repository

```bash

git clone https://github.com/pieterdeliho-hash/PaperMind.git

cd PaperMind

```



\### 2. Create Virtual Environment

```bash

python -m venv venv

venv\\Scripts\\activate  # Windows

source venv/bin/activate  # Linux/Mac

```



\### 3. Install Dependencies

```bash

pip install -r requirements.txt

```



\### 4. Configure Secrets

Create `.env` file:

```

OPENAI\_API\_KEY=sk-proj-YOUR-KEY-HERE

```



Create `.streamlit/secrets.toml`:

```toml

OPENAI\_API\_KEY = "sk-proj-YOUR-KEY-HERE"

```



\### 5. Run Locally

```bash

streamlit run streamlit\_app.py

```



\---



\## Streamlit Cloud Deployment



\### Step 1: Prepare Repository

```bash

\# Ensure .gitignore excludes secrets

cat .gitignore

\# Must include:

\# .env

\# .streamlit/secrets.toml



\# Commit and push

git add .

git commit -m "Ready for deployment"

git push origin main

```



\### Step 2: Deploy on Streamlit Cloud



1\. Go to https://share.streamlit.io/

2\. Click "New app"

3\. Configure:

&#x20;  - \*\*Repository\*\*: `pieterdeliho-hash/PaperMind`

&#x20;  - \*\*Branch\*\*: `main`

&#x20;  - \*\*Main file\*\*: `streamlit\_app.py`

&#x20;  - \*\*Python version\*\*: `3.11`



4\. Add Secrets (Settings → Secrets):

```toml

OPENAI\_API\_KEY = "sk-proj-YOUR-KEY-HERE"

```



5\. Click "Deploy!"



\### Step 3: Monitor Deployment



Watch logs for:

```

Installing dependencies from requirements.txt

Loading text index...

Loading image index...

Loading embedding models...

Connecting to OpenAI...

Multi-Modal RAG ready!

```



\---



\## Common Deployment Issues



\### Issue 1: OpenAI Client Error

\*\*Symptom\*\*: `TypeError: Client.\_\_init\_\_() got an unexpected keyword argument 'proxies'`



\*\*Fix\*\*: Pin versions in `requirements.txt`:

```

openai==1.52.0

httpx==0.27.2

```



\### Issue 2: Missing Secrets

\*\*Symptom\*\*: `OpenAI API key not found`



\*\*Fix\*\*: Add to Streamlit Cloud Secrets (NOT to git):

```toml

OPENAI\_API\_KEY = "sk-proj-..."

```



\---



\## Performance Optimization



\### Resource Limits (Streamlit Cloud)

\- \*\*RAM\*\*: 1GB (Community tier)

\- \*\*CPU\*\*: Shared

\- \*\*Storage\*\*: 50GB



\### Cost Optimization

\- \*\*Model\*\*: GPT-3.5-turbo ($0.0005/1K input, $0.0015/1K output)

\- \*\*Average query\*\*: \~3,000 tokens = $0.006

\- \*\*Monthly estimate\*\*: 1,000 queries = $6.00



\### Scaling Considerations

\- \*\*Cold start\*\*: 30-60 seconds (model loading)

\- \*\*Warm queries\*\*: 3-5 seconds

\- \*\*Rate limits\*\*: OpenAI API tier-based



\---



\## Monitoring



\### Key Metrics to Track

1\. \*\*Latency\*\*: P50, P95, P99 response times

2\. \*\*Cost\*\*: Total API spend per day/week

3\. \*\*Quality\*\*: RAGAS scores over time

4\. \*\*Usage\*\*: Queries per day



\### Logs

\- \*\*Streamlit Cloud\*\*: Manage app → Logs

\- \*\*Local\*\*: Console output with timestamps



\---



\## Security Best Practices



\### DO:

\- Use `.gitignore` for all secrets

\- Rotate API keys quarterly

\- Monitor usage for anomalies

\- Use environment variables



\### DON'T:

\- Commit `.env` or `secrets.toml`

\- Share API keys in Discord/Slack

\- Use personal keys for production

\- Hardcode credentials



\---



\## Updating Production



\### Rolling Updates

```bash

\# Make changes locally

git add .

git commit -m "Update: description"

git push origin main



\# Streamlit Cloud auto-deploys in 1-2 minutes

```



\### Manual Reboot

Streamlit Cloud → Manage app → Reboot app



\---



\## Rollback Procedure



\### If deployment breaks:

```bash

\# Locally, revert to last working commit

git log --oneline

git revert <commit-hash>

git push origin main



\# Or force push to specific commit

git reset --hard <commit-hash>

git push --force origin main

```



\---



\## Support \& Troubleshooting



\### Resources

\- \*\*Streamlit Docs\*\*: https://docs.streamlit.io/

\- \*\*OpenAI Docs\*\*: https://platform.openai.com/docs

\- \*\*RAGAS Docs\*\*: https://docs.ragas.io/



\### Contact

\- \*\*Developer\*\*: Pieter Deli Ho (pieterdeliho@gmail.com)

\- \*\*GitHub Issues\*\*: https://github.com/pieterdeliho-hash/PaperMind/issues

