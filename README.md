# Recallect

Lightweight Streamlit RAG essay reviewer. Pick a module, get a concept-grounded essay question, submit an answer, and receive a strict grade against local JSON ground truth. Groq is primary; OpenRouter is the failover.

## Local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set API keys in `.streamlit/secrets.toml` (gitignored) or env vars:

```toml
GROQ_API_KEY = "your_groq_key"
OPENROUTER_API_KEY = "your_openrouter_key"   # optional failover
```

To create an OpenRouter key via their Keys API (needs a management token):

```python
import requests

response = requests.post(
    "https://openrouter.ai/api/v1/keys",
    headers={
        "Authorization": "Bearer <OPENROUTER_MANAGEMENT_TOKEN>",
        "Content-Type": "application/json",
    },
    json={
        "name": "Recallect",
        "limit": 50,
        "limit_reset": "monthly",
        "include_byok_in_limit": True,
        "expires_at": "2027-12-31T23:59:59Z",
    },
)
print(response.text)  # paste the returned key into secrets.toml
```

Start the app:

```bash
streamlit run app.py
```

## Deploy on Render (free tier)

1. Push this repo to GitHub and create a **Web Service** on [Render](https://render.com).
2. Runtime: **Python**.
3. Build command: `pip install -r requirements.txt`
4. Start command (also in `Procfile`):
   ```
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```
5. In **Environment**, add:
   - `GROQ_API_KEY` — from [console.groq.com](https://console.groq.com)
   - `OPENROUTER_API_KEY` — from [openrouter.ai](https://openrouter.ai)
6. Deploy. Free-tier instances sleep when idle; the first request after sleep may take a minute.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI + state machine + LLM failover |
| `data/RAG_Context.json` | Module → concept/context RAG store |
| `data/reviewer/Module_*.txt` | Source study notes |
| `data/reviewer/Module_*.md` | Markdown reviewer pages (one tab per module) |
| `requirements.txt` | Sparse deps for fast builds |
| `Procfile` | Render start command |

Course PDFs under `Modules/` stay local (see `.gitignore`) and are not required at runtime — essay RAG uses `data/RAG_Context.json`, and the Module Summary Reviewer renders `data/reviewer/Module_*.md` (one full-page tab per module).
