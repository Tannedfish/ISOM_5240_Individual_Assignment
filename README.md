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

## Deploy (following the course "Streamlit Tutorial" slides)

### Part A: GitHub
1. Sign in at https://github.com → click **New** (green button, top left) to create a repository.
2. Settings: **Repository name** e.g. `ISOM5240-StoryApp`, **Visibility: Public**, **Add README: On**, **License: GNU General Public License v3.0** → **Create repository**.
3. In the repo, click **Add file → Create new file**:
   - Filename `requirements.txt` → paste its contents → **Commit changes…** → **Commit changes**.
4. Again **Add file → Create new file**:
   - Filename `app.py` → paste the whole `app.py` → **Commit changes…** → **Commit changes**.
   (Or use **Add file → Upload files** and drag both files in, then commit.)
5. Optional: open `README.md` in the repo → pencil icon → paste this README → commit.

### Part B: Streamlit Cloud
1. Go to https://share.streamlit.io → **Continue to sign-in** → **Continue with GitHub** → **Authorize streamlit**.
2. Click **Create app** (top right) → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-github-name>/ISOM5240-StoryApp`
   - **Branch:** `main`
   - **Main file path:** `app.py`  (the slides use `isom5240app.py`, but this assignment asks for `app.py`)
   - **App URL:** choose one, e.g. `isom5240-story-<yourname>`
4. Click **Deploy**. The first build installs torch and takes about 5–10 minutes. The first story then takes another 1–2 minutes while the models download.
5. Your submission link is `https://<your-app-url>.streamlit.app`.

### Part C: Add your Hugging Face token (recommended)
The models used here are public, so a token isn't strictly required. Adding one avoids Hugging Face download rate limits.
1. On huggingface.co → Settings → Access Tokens, copy your saved token or create a new one.
2. On share.streamlit.io, click the **three dots** next to your app → **Settings** → **Secrets**, then paste:
   ```
   HF_TOKEN = "hf_your_token_here"
   ```
   → **Save changes**. Streamlit passes this to the app as an environment variable, and `transformers` reads `HF_TOKEN` automatically, so `app.py` needs no changes.

### Updating the app later
Edit `app.py` on GitHub (pencil icon) → **Commit changes**. The Streamlit app updates automatically.

## Run locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| "Oh no, this app has gone over its resource limits" | Change `STORY_MODEL` in `app.py` to `HuggingFaceTB/SmolLM2-360M-Instruct` (a smaller model) and reboot the app. |
| App stuck on "Your app is in the oven" for a long time | Normal on the first build (torch is large). Wait about 10 minutes, then use ⋮ → **Reboot**. |
| No sound on iPhone/iPad | Autoplay is blocked on those devices. Press play on the audio player. |

## Testing checklist
- [ ] Photo of an animal → story mentions the animal, audio plays
- [ ] Photo of people / a place / a child's drawing
- [ ] PNG with a transparent background (the code converts it to RGB)
- [ ] Word count shown under the story is between 50 and 100
- [ ] Press the button twice in a row → a new story each time
