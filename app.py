# =========================================
# 💞 Pairfect - Where Emotions Become Art & Chemistry Finds Color
# =========================================

import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv
from openai import OpenAI
import os, re, json, base64

# ── Load environment ───────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="Pairfect 💞", page_icon="💞", layout="wide")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in .env file!")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ── CSS for animations & layout ───────────────────────────────────────────────
st.markdown(
    """
    <style>
      .center { text-align: center; }
      .gender-badge {
        display: inline-block;
        padding: 12px 26px;
        margin: 10px;
        border-radius: 40px;
        font-size: 22px;
        font-weight: bold;
        color: white;
        animation: glow 2s infinite alternate;
        text-shadow: 0px 0px 8px rgba(255,255,255,0.8);
      }
      .male {
        background: linear-gradient(90deg, #007BFF, #00C6FF);
        box-shadow: 0 0 25px rgba(0,123,255,0.6);
      }
      .female {
        background: linear-gradient(90deg, #ff4b6e, #ff80a0);
        box-shadow: 0 0 25px rgba(255,105,180,0.6);
      }
      .mixed {
        background: linear-gradient(90deg, #ff4b6e, #00bfff);
        box-shadow: 0 0 30px rgba(255,105,180,0.5);
      }
      @keyframes glow {
        0% { transform: scale(1); opacity: 0.85; }
        100% { transform: scale(1.07); opacity: 1; }
      }
      .poem-box {
        text-align: center;
        margin: 24px auto;
        width: min(780px, 90%);
        background: rgba(255, 240, 245, 0.65);
        border: 2px solid #ffb6c1;
        border-radius: 20px;
        padding: 20px 24px;
        font-size: 17px;
        font-style: italic;
        color: #333;
        transition: transform 0.35s ease, box-shadow 0.35s ease;
      }
      .poem-box:hover {
        transform: scale(1.03);
        box-shadow: 0px 0px 20px rgba(255,182,193,0.85);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Emotion Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
    )
emotion_model = load_emotion_model()

def analyze_emotion(text: str):
    if not text.strip():
        return {"emotions": {}, "top_emotion": "neutral"}
    res = emotion_model(text)[0]
    emotions = {r["label"]: round(r["score"] * 100, 2) for r in res}
    top = max(emotions, key=emotions.get)
    if any(w in text.lower() for w in ["calm", "peace", "relaxed", "serene"]):
        top = "calm"
    return {"emotions": emotions, "top_emotion": top}

# ── Vision Analysis ───────────────────────────────────────────────────────────
def analyze_couple_image(photo_bytes):
    """Detect gender, mood, appearance, outfit with correction."""
    try:
        img_b64 = base64.b64encode(photo_bytes).decode("utf-8")
        prompt = """
        Analyze this couple photo and return JSON:
        {
          "count": number of people,
          "people": [
            {"gender": "male/female/unknown", "mood": "...", "appearance": "...", "outfit": "..."}
          ]
        }
        """
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}},
                ]}
            ],
        )
        match = re.search(r"\{.*\}", r.choices[0].message.content, re.S)
        data = json.loads(match.group(0)) if match else {"count": 0, "people": []}

        # Gender correction logic (simple hint-based)
        if data.get("count", 0) == 2:
            p1, p2 = data["people"]
            desc = f"{p1['appearance']} {p1['outfit']} {p2['appearance']} {p2['outfit']}".lower()
            if p1["gender"] == p2["gender"]:
                male_hints = any(k in desc for k in ["shirt", "beard", "short hair", "kurta"])
                female_hints = any(k in desc for k in ["saree", "lipstick", "long hair", "dress"])
                if male_hints and not female_hints:
                    p1["gender"], p2["gender"] = "male", "male"
                elif female_hints and not male_hints:
                    p1["gender"], p2["gender"] = "female", "female"
        return data

    except Exception as e:
        st.error(f"Vision analysis failed: {e}")
        return {"count": 0, "people": []}

# ── AI + Art Helpers ──────────────────────────────────────────────────────────
def get_ai_summary(u1, u2):
    p = f"""
You are Pairfect AI – a romantic coach.
Analyze chemistry between {u1['name']} ({u1['gender']}) and {u2['name']} ({u2['gender']}).
Provide:
1. Overview
2. Compatibility Score XX/100
3. Why they connect or differ
4. Strengths and growth tip.
"""
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": p}])
    return r.choices[0].message.content.strip()

def extract_score(text):
    m = re.search(r"(\d{1,3})\s*/\s*100", text) or re.search(r"(\d{1,3})%", text)
    return min(max(int(m.group(1)), 0), 100) if m else 75

def generate_art_prompt(u1, u2):
    return f"""
Dreamlike cinematic digital art of {u1['name']} ({u1['gender']}) and {u2['name']} ({u2['gender']}). 
{u1['name']} is {u1['appearance']} wearing {u1['outfit']}. 
{u2['name']} is {u2['appearance']} wearing {u2['outfit']}. 
Mood: {u1['mood']} and {u2['mood']}. 
A glowing romantic setting that represents their bond.
"""

def generate_poem(u1, u2):
    prompt = f"""
Write a poetic 'Pairfect Thought' (4–6 lines) about:
{u1['name']} and {u2['name']} — their moods {u1['mood']} & {u2['mood']}, appearances {u1['appearance']} / {u2['appearance']}, outfits {u1['outfit']} / {u2['outfit']}.
"""
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip()

def generate_art(prompt: str):
    try:
        r = client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024")
        return r.data[0].url
    except Exception as e:
        st.error(str(e))
        return None

def love_coach_reply(user_msg, ctx):
    system = "You are 'Pairfect Love Coach' — warm, empathetic, and insightful."
    msg = f"Couple: {ctx['u1_name']} & {ctx['u2_name']} ({ctx['score']}%)\nSummary: {ctx['summary']}\nUser: {user_msg}"
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system}, {"role": "user", "content": msg}])
    return r.choices[0].message.content.strip()

# ❤️ Love Meter
def heart_meter(score):
    color = "#ff4b6e" if score < 80 else "gold"
    opacity = score / 100
    st.markdown(f"""
    <div class='center'>
      <svg viewBox="0 0 200 200" width="120" height="120">
        <path d="M100 180 C 20 100, 80 20, 100 60 C 120 20, 180 100, 100 180 Z"
          fill="{color}" opacity="{opacity}" stroke="pink" stroke-width="3">
          <animate attributeName="opacity" values="0;{opacity};0.6;{opacity}" dur="2s" repeatCount="indefinite"/>
        </path>
      </svg>
      <p style="font-size:22px;color:{color};font-weight:bold;">Compatibility Score: {score}%</p>
    </div>
    """, unsafe_allow_html=True)

def describe_for_art(img_bytes: bytes) -> str:
    """Create a compact, vivid prompt from a couple photo."""
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = (
        "Look at the couple photo and write ONE vivid sentence I can use as an image prompt. "
        "Mention hair, outfits, pose, setting, vibe; keep faces recognizable; no JSON."
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ],
        }],
    )
    return (r.choices[0].message.content or "").strip()

# ── State Initialization ──────────────────────────────────────────────────────
for k, v in {"page": "Compatibility & Art", "content": None, "ctx": None, "photo_hash": None,
             "gender_label": None, "chat_history": [], "vision_result": None}.items():
    st.session_state.setdefault(k, v)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='center' style='color:#ff4b6e;'>💞 Pairfect</h1>", unsafe_allow_html=True)
st.markdown("<p class='center' style='color:gray;'>Where Emotions Become Art & Chemistry Finds Color</p>", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("🎨 Compatibility & Art", use_container_width=True):
        st.session_state.page = "Compatibility & Art"
with col2:
    if st.button("🧠 Love Coach Chat", use_container_width=True):
        st.session_state.page = "Love Coach Chat"
with col3:
    if st.button("🎧 Mood Music", use_container_width=True):
        st.session_state.page = "Mood Music"
with col4:
    if st.button("🖼️ Love Art Gallery", use_container_width=True):
        st.session_state.page = "Love Art Gallery"
with col5:
    if st.button("ℹ️ About Pairfect", use_container_width=True):
        st.session_state.page = "About Pairfect"

st.markdown("<hr style='border:1px solid pink;'>", unsafe_allow_html=True)

# ── Compatibility & Art Page ──────────────────────────────────────────────────
if st.session_state.page == "Compatibility & Art":
    st.subheader("📸 Upload or Capture Your Couple Photo")
    photo = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])
    snap = st.camera_input("Or Take a Photo 💕")
    img = photo or snap
    if not img:
        for key in ["content", "ctx", "gender_label", "vision_result"]:
            st.session_state.pop(key, None)
        st.stop()

    h = hash(img.getvalue())
    if st.session_state.photo_hash != h:
        st.session_state.photo_hash = h

        # 🧹 Reset all session data for a fresh analysis
        st.session_state.chat_history = []   # Clear previous Love Coach chat
        st.session_state.ctx = None          # Reset context
        st.session_state.content = None      # Reset summary, art, poem, etc.
        st.session_state.gender_label = None # Reset gender badge
        
        with st.spinner("Analyzing your photo with Vision AI..."):
            result = analyze_couple_image(img.getvalue())
        st.session_state.vision_result = result
    else:
        result = st.session_state.vision_result

    if not result or result["count"] != 2:
        st.warning("⚠️ Please upload a photo with exactly two people.")
        st.stop()

    p1, p2 = result["people"]
    g1, g2 = p1["gender"], p2["gender"]
    pair_type = "mixed" if g1 != g2 else "male" if g1 == "male" else "female"
    label = "👫 Male + Female" if pair_type == "mixed" else "👬 Male + Male" if pair_type == "male" else "👭 Female + Female"
    st.session_state.gender_label = f"<div class='center'><span class='gender-badge {pair_type}'>{label}</span></div>"
    st.markdown(st.session_state.gender_label, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        u1 = {"gender": g1, "name": st.text_input("Your Name", key="u1n"), "desc": st.text_area("Your Personality", key="u1d"),
              "mood": st.text_input("Detected Mood", value=p1["mood"], key="u1m"),
              "appearance": st.text_input("Detected Appearance", value=p1["appearance"], key="u1a"),
              "outfit": st.text_input("Detected Outfit", value=p1["outfit"], key="u1o"),
              "interests": st.text_input("Your Interests", key="u1i")}
    with c2:
        u2 = {"gender": g2, "name": st.text_input("Partner’s Name", key="u2n"), "desc": st.text_area("Partner Personality", key="u2d"),
              "mood": st.text_input("Detected Mood", value=p2["mood"], key="u2m"),
              "appearance": st.text_input("Detected Appearance", value=p2["appearance"], key="u2a"),
              "outfit": st.text_input("Detected Outfit", value=p2["outfit"], key="u2o"),
              "interests": st.text_input("Partner Interests", key="u2i")}

    if st.button("✨ Generate Pairfect Analysis", use_container_width=True):
        if not (u1["name"] and u2["name"] and u1["desc"] and u2["desc"]):
            st.warning("Please fill in both names and descriptions!")
        else:
            u1["emotion"], u2["emotion"] = analyze_emotion(u1["desc"]), analyze_emotion(u2["desc"])
            with st.spinner("Analyzing your chemistry 💫"):
                summary = get_ai_summary(u1, u2)
            score = extract_score(summary)
            art_prompt = generate_art_prompt(u1, u2)
            art_url = generate_art(art_prompt)
            poem = generate_poem(u1, u2)
            st.session_state.content = {
                "summary": summary,
                "score": score,
                "art_prompt": art_prompt,
                "art_url": art_url,
                "poem": poem,
                "u1_emotion": u1["emotion"]["emotions"],
                "u2_emotion": u2["emotion"]["emotions"],
            }
            st.session_state.ctx = {"u1_name": u1["name"], "u2_name": u2["name"], "score": score, "summary": summary}
            st.success("✅ Pairfect Analysis Complete!")

    if st.session_state.get("content"):
        c = st.session_state.content
        st.markdown("---")
        st.subheader("💞 AI Compatibility Insights")
        st.write(c["summary"])
        heart_meter(c["score"])
        st.subheader("🎨 Art Prompt")
        st.write(c["art_prompt"])
        if c["art_url"]:
            st.markdown(f"<div class='center'><img src='{c['art_url']}' width='500' "
                        "style='border-radius:20px;box-shadow:0 8px 24px rgba(255,105,180,0.3);'/></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='poem-box'><h4 style='color:#ff4b6e;'>📝 Poetic 'Pairfect Thought'</h4><p>{c['poem']}</p></div>", unsafe_allow_html=True)

# ── Love Coach Chat ───────────────────────────────────────────────────────────
if st.session_state.page == "Love Coach Chat":
    st.subheader("🧠 Pairfect Love Coach")
    ctx = st.session_state.ctx
    if not ctx:
        st.info("💌 Please run 'Generate Pairfect Analysis' in Compatibility & Art first!")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_msg = st.chat_input("Ask the Love Coach anything about your connection…")

        if user_msg:
            # Show the user's message instantly
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            # Generate and display assistant reply
            with st.chat_message("assistant"):
                with st.spinner("💞 Pairfect Love Coach is thinking..."):
                    reply = love_coach_reply(user_msg, ctx)
                st.markdown(reply)

            # Save assistant reply to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ── Mood Music Tab ────────────────────────────────────────────────────────────
if st.session_state.page == "Mood Music":
    st.subheader("🎧 Your Personalized Romantic Vibes")

    c = st.session_state.get("content")
    if not c:
        st.info("💌 Please complete Compatibility & Art to unlock Mood Music.")
    else:
        st.markdown("### 🎵 Choose Your Mood & Language")

        mood = st.selectbox(
            "Select the vibe that matches your love today 💞",
            [
                "Love",
                "Joy",
                "Calm",
                "Sadness",
                "Anger",
                "Nostalgia",
                "Party",
                "Devotional"
            ],
            index=0
        )

        language = st.selectbox(
            "Select your preferred language 🎤",
            ["Tamil", "English", "Malayalam", "Hindi"],
            index=0
        )

        # Spotify playlists mapping
        playlists = {
            "Tamil": {
                "Love": "5AB5tIUaoGyIsa5U2vY8Z9",       
                "Joy": "5H8p8FuYHKWCW5nboTHhqg",       
                "Calm": "0TOng1FTBaa6bHJcfack1S",     
                "Sadness": "0AyOLKzLZZmlliok7bu1mp",
                "Anger": "3p8ejB7BscAVmEdyK7AtXx",
                "Nostalgia": "5igRa5EOOSF1U6ZdnkwqXi",
                "Party": "2rDck89vUaM7SoGih1gzsU",
                "Devotional": "4TAMPUxAKdRJc32ZFaVkGy",
            },
            "English": {
                "Love": "5QOrHPIzTFh80WhHmbOcCp",
                "Joy": "0jrlHA5UmxRxJjoykf7qRY",
                "Calm": "4kOdiP5gbzocwxQ8s2UTOF",
                "Sadness": "25ZzkJkOuYir9kHr2CqwPQ",
                "Anger": "67STztGl7srSMNn6hVYPFR",
                "Nostalgia": "3DEdLxmZTeLBIpnFedtQDa",
                "Party": "3y96TXf7zKJTv48OlP4xEB",
                "Devotional": "0SKDsYUwb8jlqKADTJAiBY",
            },
            "Malayalam": {
                "Love": "75mxLeOLm7uJRSqebqZQL9",
                "Joy": "37i9dQZF1DWTYKFynxp6Fs",
                "Calm": "5JgrCNuRqcIgXfiN3JTclm",
                "Sadness": "05975N5TZBOmYsHFnqxXWZ",
                "Anger": "09aHjp6uoqqGsthJwsbxvj",
                "Nostalgia": "1tOdyNZ0SKvPU0cjCPXRte",
                "Party": "37i9dQZF1DX7ko3EzbLi5w",
                "Devotional": "62RrHfsbwsNg4AfnnD1w0B",
            },
            "Hindi": {
                "Love": "6kaEWP7NTNRVWnFbJT3MCh",
                "Joy": "4Xs08zUmeoEBTacEv2vA6Y",
                "Calm": "1Dk9SeguLL5qTnjfyX5VnZ",
                "Sadness": "189Sow1xr7R94oSKs4kISc",
                "Anger": "3JNWpteYvH3ynMcyPcvxfx",
                "Nostalgia": "37i9dQZF1DWXRgBZj2DeRk",
                "Party": "1SX3oHTD0iRZM4c7TXZKL9",
                "Devotional": "0fOeg4UBBvQTfxut6zdbfm",
            }
        }

        playlist_id = playlists.get(language, {}).get(mood, None)

        if playlist_id:
            st.markdown(f"<h4 style='color:#ff4b6e;'>💓 {mood} Vibes ({language})</h4>", unsafe_allow_html=True)
            st.markdown(f"""
                <iframe src="https://open.spotify.com/embed/playlist/{playlist_id}"
                width="100%" height="400" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
            """, unsafe_allow_html=True)
        else:
            st.warning("No playlist available for this combination yet 💔")

        # Romantic closing tagline
        st.markdown(
            "<p style='text-align:center; font-size:18px; color:#ff4b6e; margin-top:20px;'>✨ Let the rhythm of love play on, and hearts dance to its tune... 💖</p>",
            unsafe_allow_html=True,
        )

# ── Love Art Gallery Tab ───────────────────────────────────────────────────────
if st.session_state.page == "Love Art Gallery":
    st.subheader("🖼️ Love Art Gallery – Memories in Motion 💞")

    # Secure password from environment
    GALLERY_PASS = os.getenv("LOVE_GALLERY_PASS")

    st.markdown(
        """
        <div style='background:rgba(255,240,245,0.7);padding:15px;border-radius:15px;'>
        <p style='font-size:16px;color:#333;'>
        🔒 <b>This feature is password-protected</b> to manage API usage and cost. <br>
        If you'd like access, please send an email to 
        <a href="mailto:farhunhazard@gmail.com" style="color:#ff4b6e;font-weight:bold;">farhunhazard@gmail.com</a> 
        explaining your purpose for using the <b>Love Art Gallery</b> feature.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    password = st.text_input("Enter Access Password to Unlock Gallery 🔑", type="password")

    if password != GALLERY_PASS:
        st.warning("🚫 Access Denied. Please enter the correct password to continue.")
        st.stop()
    else:
        st.success("✅ Access Granted! Welcome to the Love Art Gallery 💞")

    # Ensure Pairfect Analysis is done first
    if not st.session_state.get("content"):
        st.info("💌 Please run 'Generate Pairfect Analysis' in Compatibility & Art first!")
        st.stop()

    st.markdown("### 🎭 Every bond tells a story — let AI turn your love into digital art.")

    uploaded_files = st.file_uploader(
        "📸 Upload exactly 3 couple photos to transform into romantic artworks",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) != 3:
            st.warning("Please upload **exactly 3 photos** for the full Love Art Gallery experience 💞")
        else:
            st.success("✅ Perfect! You've uploaded 3 beautiful memories. Let’s turn them into AI art...")

            art_images = []

            for idx, img_file in enumerate(uploaded_files, start=1):
                with st.spinner(f"🎨 Creating your dreamy AI art #{idx} for {img_file.name}..."):
                    try:
                        img_bytes = img_file.read()
                        base_desc = describe_for_art(img_bytes)
                        style = (
                            " — render as a cinematic romantic digital painting with warm, soft light, "
                            "gentle bokeh, pastel glow, painterly brush strokes; keep faces recognizable."
                        )
                        art_prompt = (base_desc or "A smiling couple in a tender pose") + style

                        # Primary: gpt-image-1
                        art_result = client.images.generate(
                            model="gpt-image-1",
                            prompt=art_prompt,
                            size="1024x1024",
                            n=1
                        )
                        art_b64 = art_result.data[0].b64_json
                        art_bytes = base64.b64decode(art_b64)
                        art_images.append(f"data:image/png;base64,{base64.b64encode(art_bytes).decode()}")

                    except Exception:
                        # Fallback: DALL·E 3 (URL)
                        try:
                            art_result = client.images.generate(
                                model="dall-e-3",
                                prompt=art_prompt,
                                size="1024x1024",
                                n=1
                            )
                            art_images.append(art_result.data[0].url)
                        except Exception as e2:
                            st.error(f"Failed to generate art for {img_file.name}: {e2}")

            # --- CSS-ONLY SLIDESHOW (no JS) ---
            if art_images:
                st.markdown("<h4 style='text-align:center;color:#ff4b6e;'>💞 Your AI-Generated Love Art Slideshow</h4>", unsafe_allow_html=True)

                # CSS Keyframe Slideshow
                html = f"""
                <div class="slideshow-container">
                    {''.join([f"<div class='slide fade' style='background-image: url({img});'></div>" for img in art_images])}
                </div>

                <style>
                .slideshow-container {{
                    position: relative;
                    width: 100%;
                    max-width: 600px;
                    height: 600px;
                    margin: 0 auto;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 8px 24px rgba(255,105,180,0.4);
                }}
                .slide {{
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    background-size: cover;
                    background-position: center;
                    opacity: 0;
                    animation: fade 18s infinite;
                }}
                .slide:nth-child(1) {{ animation-delay: 0s; }}
                .slide:nth-child(2) {{ animation-delay: 6s; }}
                .slide:nth-child(3) {{ animation-delay: 12s; }}

                @keyframes fade {{
                    0% {{ opacity: 0; }}
                    10% {{ opacity: 1; }}
                    30% {{ opacity: 1; }}
                    40% {{ opacity: 0; }}
                    100% {{ opacity: 0; }}
                }}
                </style>
                """
                st.markdown(html, unsafe_allow_html=True)

                st.markdown(
                    "<p style='text-align:center; color:#ff4b6e; font-size:18px; margin-top:25px;'>"
                    "✨ Love captured, colors revealed — your art speaks the language of your heart 💖"
                    "</p>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Upload 3 photos to see the magic of love-infused AI art! 🎨")

# ── About Pairfect Tab ─────────────────────────────────────────────────────────
if st.session_state.page == "About Pairfect":
    st.markdown("<h1 class='center' style='color:#ff4b6e;'>🌸 About Pairfect</h1>", unsafe_allow_html=True)
    st.markdown("<p class='center' style='color:gray;'>Where Emotion Meets Innovation, and AI Paints the Language of Love 💫</p>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='background:rgba(255,240,245,0.5);border-radius:20px;padding:25px;box-shadow:0 4px 18px rgba(255,182,193,0.4);'>
        <h3 style='color:#ff4b6e;'>💞 Vision & Purpose</h3>
        <p style='font-size:17px;color:#333;'>
        <b>Pairfect</b> redefines how relationships are understood and celebrated using the emotional intelligence of AI. 
        It’s more than a compatibility app — it’s a digital companion that analyzes couple dynamics, transforms chemistry into art, 
        generates mood-based playlists, and provides personalized romantic coaching — all within one seamless experience.
        </p>

        <h3 style='color:#ff4b6e;'>🚀 Functionality & AI Integration</h3>
        <ul style='font-size:16px;line-height:1.6;color:#333;'>
          <li><b>AI Compatibility Engine</b> – Uses GPT-4o and HuggingFace emotional models to analyze personalities, emotions, and visual cues from photos.</li>
          <li><b>Vision AI</b> – Detects mood, appearance, and gender to personalize compatibility insights and art prompts.</li>
          <li><b>Love Coach Chat</b> – A GPT-powered guide that provides real-time advice, conflict resolution, and growth tips.</li>
          <li><b>Mood Music</b> – Integrates Spotify playlists dynamically based on emotional tone and preferred language.</li>
          <li><b>Love Art Gallery</b> – Converts couple photos into AI-generated romantic artworks using DALL·E 3 and GPT-image-1.</li>
        </ul>

        <h3 style='color:#ff4b6e;'>🎨 Innovation & Creativity</h3>
        <p style='font-size:17px;color:#333;'>
        What makes Pairfect stand out is its <b>fusion of emotion analysis, visual storytelling, and generative art</b>. 
        While traditional compatibility apps rely on surveys or astrology, Pairfect brings <b>data-driven emotional intelligence</b> 
        and <b>AI artistry</b> to build a personalized love narrative.  
        The seamless blend of <b>psychological profiling, music curation, and AI art</b> creates a “wow factor” that appeals both emotionally and intellectually.
        </p>

        <h3 style='color:#ff4b6e;'>💡 Real-World Impact & Future Scope</h3>
        <p style='font-size:17px;color:#333;'>
        Pairfect can evolve into an <b>AI-powered relationship wellness platform</b> offering:
        </p>
        <ul style='font-size:16px;line-height:1.6;color:#333;'>
          <li>💬 <b>Couple Therapy-as-a-Service</b> – Personalized insights to help partners understand each other better.</li>
          <li>🎁 <b>AI-Generated Memories</b> – Monthly art or poem drops for anniversaries or milestones.</li>
          <li>📊 <b>Emotional Analytics Dashboard</b> – Track relationship growth and shared moods over time.</li>
          <li>💍 <b>Integration with Dating Platforms</b> – Offer “emotional compatibility” scoring for new matches.</li>
          <li>🌎 <b>Community & NFT Marketplace</b> – Tokenize love art and create collectible digital keepsakes.</li>
        </ul>

        <h3 style='color:#ff4b6e;'>💻 Technical Execution</h3>
        <p style='font-size:17px;color:#333;'>
        Pairfect uses <b>multi-model orchestration</b> with OpenAI’s GPT-4o, HuggingFace emotion analysis, and DALL·E 3 for creative rendering.  
        It is built using <b>Streamlit</b> for UI scalability and maintains state with <b>session-based caching</b> for smooth navigation.  
        The solution ensures <b>low-latency inference</b>, <b>secure photo handling</b>, and <b>modular extensibility</b> for adding more languages or features effortlessly.
        </p>

        <h3 style='color:#ff4b6e;'>🧭 User Experience & Accessibility</h3>
        <p style='font-size:17px;color:#333;'>
        The interface was designed with simplicity and inclusivity at its heart. From AI-guided photo insights to an emotionally responsive UI, 
        users of all backgrounds — whether tech-savvy or beginners — can experience an intuitive and joyful journey.  
        Multi-language music support ensures <b>cultural inclusivity</b>, making love truly borderless. 💕
        </p>

        <h3 style='color:#ff4b6e;'>💰 Why It’s a Hackathon Winner & Investor-Ready</h3>
        <p style='font-size:17px;color:#333;'>
        Pairfect embodies the hackathon spirit — combining creativity, human emotion, and AI precision.  
        Its <b>modular AI stack</b> and <b>emotion-driven storytelling</b> make it scalable into a mainstream product.  
        With growing demand for <b>AI wellness</b> and <b>relationship tech</b>, Pairfect is poised to lead a new wave of emotionally intelligent digital experiences.  
        Investors can monetize via <b>premium subscriptions, AI art sales, partnership APIs</b>, and <b>event-based personalization</b>.  
        It’s not just an app — it’s a movement to make technology feel human again. 💖
        </p>

        <h3 style='color:#ff4b6e;'>🔮 Future Advancements & Roadmap</h3>
        <p style='font-size:17px;color:#333;'>
        The journey of Pairfect doesn’t stop here — it’s just beginning. Our roadmap focuses on deepening emotional intelligence, personalization, and global scalability:
        </p>
        <ul style='font-size:16px;line-height:1.6;color:#333;'>
          <li>🦾 <b>AI Personality Engine 2.0</b> – Integrate real-time sentiment tracking and predictive emotional modeling for dynamic relationship insights.</li>
          <li>🧬 <b>Behavioral Matching Algorithm</b> – Combine voice tone, text semantics, and visual expression for deeper compatibility analysis.</li>
          <li>💬 <b>Voice-Interactive Love Coach</b> – Transform the Love Coach into an AI companion powered by emotion-aware voice synthesis.</li>
          <li>🪄 <b>Augmented Reality (AR) Art</b> – Bring AI-generated love art to life through AR filters and 3D memories.</li>
          <li>🌐 <b>Decentralized Love Vault</b> – Secure couple data and digital art using blockchain-based privacy and ownership layers.</li>
          <li>📱 <b>Mobile App Launch</b> – Launch Pairfect on iOS and Android with enhanced local caching and offline support.</li>
          <li>❤️ <b>Partnerships</b> – Collaborate with wedding planners, wellness apps, and dating platforms to integrate Pairfect’s emotional AI as a plugin service.</li>
        </ul>
        <p style='font-size:17px;color:#333;'>
        Each milestone brings Pairfect closer to becoming the world’s first <b>Emotionally Intelligent Relationship Ecosystem</b> — 
        where technology doesn’t just understand love; it <b>feels it</b>.
        </p>

        <h3 style='color:#ff4b6e;'>🌠 Closing Note</h3>
        <p style='font-size:18px;color:#333;text-align:center;'>
        "Pairfect isn’t just built for love — it’s built with love.  
        A union of code, creativity, and connection — redefining how technology understands the human heart."
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Built with ❤️ using Streamlit, OpenAI Vision, and Hugging Face.</p>", unsafe_allow_html=True)


