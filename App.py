
import io
import re

import streamlit as st
import torch
from gtts import gTTS
from PIL import Image
from transformers import pipeline

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MIN_STORY_WORDS = 50
MAX_STORY_WORDS = 100
MAX_NEW_TOKENS = 180          # roughly 130 words - enough room, then we trim to 100
MAX_GENERATION_ATTEMPTS = 3   # retry if the story comes out too short

ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


# ---------------------------------------------------------------------------
# Model loading (cached so the models load only once per server, not per click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Waking up the story robot... (first time takes about a minute)")
def load_caption_pipeline():
    """Load the BLIP image-captioning pipeline."""
    return pipeline("image-to-text", model=CAPTION_MODEL)


@st.cache_resource(show_spinner="Opening the magic story book...")
def load_story_pipeline():
    """Load the instruction-tuned text-generation pipeline.

    bfloat16 halves memory use (about 1 GB instead of 2 GB), which keeps the
    app within Streamlit Community Cloud's memory limit.
    """
    return pipeline("text-generation", model=STORY_MODEL, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Stage 1: image -> caption
# ---------------------------------------------------------------------------
def generate_caption(image: Image.Image) -> str:
    """Return a short English description of the uploaded image."""
    captioner = load_caption_pipeline()
    result = captioner(image)
    caption = result[0]["generated_text"].strip()
    return caption


# ---------------------------------------------------------------------------
# Stage 2: caption -> story
# ---------------------------------------------------------------------------
def build_story_prompt(caption: str) -> list:
    """Build chat messages asking the model for a gentle children's story."""
    system_message = (
        "You are a friendly storyteller for children aged 3 to 10. "
        "You write happy, gentle, safe stories using simple words and short sentences. "
        "Never include anything scary, violent or sad."
    )
    user_message = (
        f"Write a short bedtime story of about 70 words based on this picture: "
        f'"{caption}". Give the main character a fun name, include a small '
        f"adventure, and end with a happy ending. Write only the story, with no title."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def count_words(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


def trim_to_word_limit(text: str, max_words: int = MAX_STORY_WORDS) -> str:
    """Cut the story to at most `max_words`, ending on a complete sentence."""
    if count_words(text) <= max_words:
        return text

    # Keep whole sentences while we stay under the limit
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept_sentences = []
    for sentence in sentences:
        candidate = " ".join(kept_sentences + [sentence])
        if count_words(candidate) > max_words:
            break
        kept_sentences.append(sentence)

    trimmed = " ".join(kept_sentences)
    # Fallback: if even the first sentence is too long, hard-cut the words
    if not trimmed:
        trimmed = " ".join(text.split()[:max_words]).rstrip(",;:") + "."
    return trimmed


def clean_story_text(raw_text: str) -> str:
    """Remove titles, markdown symbols and extra whitespace from the model output."""
    text = raw_text.strip()
    # Drop a leading title line such as "Title: ..." or "**The Brave Cat**"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and (lines[0].lower().startswith("title") or lines[0].startswith(("*", "#"))):
        lines = lines[1:]
    text = " ".join(lines)
    text = re.sub(r"[*#_]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generate_story(caption: str) -> str:
    """Expand the caption into a 50-100 word children's story."""
    story_generator = load_story_pipeline()
    messages = build_story_prompt(caption)

    best_story = ""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        output = story_generator(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        # With chat input, the pipeline returns the whole conversation;
        # the last message is the assistant's reply (our story).
        raw_story = output[0]["generated_text"][-1]["content"]
        story = trim_to_word_limit(clean_story_text(raw_story))

        if count_words(story) > count_words(best_story):
            best_story = story
        if count_words(story) >= MIN_STORY_WORDS:
            break

    return best_story


# ---------------------------------------------------------------------------
# Stage 3: story -> audio
# ---------------------------------------------------------------------------
def text_to_speech(text: str) -> bytes:
    """Convert the story to MP3 audio bytes using Google Text-to-Speech."""
    audio_buffer = io.BytesIO()
    gTTS(text=text, lang="en", slow=False).write_to_fp(audio_buffer)
    return audio_buffer.getvalue()


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
def show_header():
    """Page title and friendly instructions for kids (and their parents)."""
    st.set_page_config(page_title="Magic Picture Stories", page_icon="📖", layout="centered")
    st.title("📖 Magic Picture Stories")
    st.markdown(
        "### Show me a picture and I'll tell you a story! 🐻🚀🌈\n"
        "1. Pick a picture 🖼️  \n"
        "2. Press the **Tell me a story!** button ✨  \n"
        "3. Read along and listen 🎧"
    )


def main():
    """Run the Streamlit app."""
    show_header()

    uploaded_file = st.file_uploader("🖼️ Choose a picture", type=ALLOWED_IMAGE_TYPES)
    if uploaded_file is None:
        st.info("👆 Upload a photo or drawing to begin!")
        return

    # Convert to RGB so PNGs with transparency also work with BLIP
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your picture", width="stretch")

    if not st.button("✨ Tell me a story!", type="primary", width="stretch"):
        return

    try:
        with st.status("Making your story...", expanded=True) as status:
            st.write("👀 Looking at your picture...")
            caption = generate_caption(image)
            st.write(f"I see: *{caption}*")

            st.write("✍️ Writing your story...")
            story = generate_story(caption)

            st.write("🎤 Getting ready to read it to you...")
            audio_bytes = text_to_speech(story)

            status.update(label="Your story is ready! 🎉", state="complete", expanded=False)
    except Exception as error:  # show a friendly message instead of a crash
        st.error("Oops! The story robot got a little tired. Please try again. 🤖💤")
        st.caption(f"Technical details: {error}")
        return

    st.subheader("🌟 Your Story")
    st.markdown(f"<div style='font-size:1.3rem; line-height:1.7'>{story}</div>", unsafe_allow_html=True)
    st.caption(f"{count_words(story)} words")

    st.subheader("🎧 Listen")
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    st.balloons()


if __name__ == "__main__":
    main()