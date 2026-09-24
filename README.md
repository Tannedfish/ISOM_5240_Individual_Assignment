# Magic Picture Stories: ISOM5240 Storytelling App

A Streamlit app for kids aged 3 to 10. Upload a picture and the app describes it, writes a 50–100 word story about it, and reads the story aloud.

## Pipeline

| Stage | Tool | Function in `app.py` |
|---|---|---|
| Image → caption | `Salesforce/blip-image-captioning-base` (HF `image-to-text` pipeline) | `generate_caption()` |
| Caption → story | `Qwen/Qwen2.5-0.5B-Instruct` (HF `text-generation` pipeline) | `generate_story()` |
| Story → audio | gTTS (Google Text-to-Speech) | `text_to_speech()` |

Design notes:
- The models are cached with `@st.cache_resource`, so they load once per server instead of on every click.
- The story model is loaded in `bfloat16`, which halves its memory use so the app fits within Streamlit Community Cloud's RAM limit.
- The 50–100 word requirement is enforced in code: `trim_to_word_limit()` cuts the story at a sentence boundary, and `generate_story()` retries up to 3 times if the story is shorter than 50 words.
- The system prompt keeps the stories gentle and not scary, which suits young children.

## Deploy to Streamlit Community Cloud

1. **Create a GitHub repo.** Go to github.com → New repository (make it **Public**), then upload `app.py`, `requirements.txt` and this `README.md` to the repo root.
2. **Sign in to Streamlit.** Go to share.streamlit.io and sign in with GitHub.
3. **Create the app.** Click **Create app** → **Deploy a public app from GitHub**, then set:
   - Repository: `<your-username>/<repo-name>`
   - Branch: `main`
   - Main file path: `app.py`
   - Advanced settings → Python version: **3.12**
4. Click **Deploy**. The first build takes about 5–10 minutes to install the packages. The first story takes another 1–2 minutes because the models download on first use. Stories after that take about 10–30 seconds.
5. Copy the URL (`https://<something>.streamlit.app`) and submit it.

### Optional: use your Hugging Face token
All three models are public, so no token is needed. If the logs show Hugging Face rate-limit errors (HTTP 429), open your HF account → Settings → Access Tokens and create a **Read** token. Then, in Streamlit Cloud, go to the app → Settings → Secrets and add:
```
HF_TOKEN = "hf_xxxxxxxx"
```
Streamlit exposes secrets as environment variables, and `transformers` reads `HF_TOKEN` automatically.

## Run locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| "Oh no, this app has gone over its resource limits" | Change `STORY_MODEL` in `app.py` to `HuggingFaceTB/SmolLM2-360M-Instruct` (a smaller model) and reboot the app. |
| Build fails while installing torch | Delete the `--extra-index-url` line from `requirements.txt` and redeploy. The build is slower but uses the standard PyPI torch. |
| No sound on iPhone/iPad | Autoplay is blocked on those devices. Press play on the audio player. |

## Testing checklist
- [ ] Photo of an animal → story mentions the animal, audio plays
- [ ] Photo of people / a place / a child's drawing
- [ ] PNG with a transparent background (the code converts it to RGB)
- [ ] Word count shown under the story is between 50 and 100
- [ ] Press the button twice in a row → a new story each time
