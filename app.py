"""
app.py  —  Truthscope  (Text + Video + Audio multimodal fake-news detector)
────────────────────────────────────────────────────────────────────────────────
UI changes vs previous version
  • Audio uploader added (WAV)
  • Modality badge now covers all 7 combinations
  • Demo file picker shows actual filenames from datasets/ with folder label
  • Result card shows per-modality signal including audio
  • Raw scores panel extended with modality field
────────────────────────────────────────────────────────────────────────────────
"""

import os
import glob
import tempfile
import streamlit as st

from inference import predict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Truthscope — Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;1,400;1,600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1 { font-family: 'Fraunces', serif; font-style: italic; letter-spacing: -1px; }
.verdict-fake { background:#fff0f0; border:1px solid #ff4b4b; border-radius:8px; padding:1.2rem 1.5rem; margin-top:1rem; }
.verdict-real { background:#f0fff4; border:1px solid #21c55d; border-radius:8px; padding:1.2rem 1.5rem; margin-top:1rem; }
.verdict-label { font-size:0.75rem; letter-spacing:0.12em; text-transform:uppercase; opacity:0.6; margin-bottom:0.25rem; }
.verdict-value { font-family:'Fraunces',serif; font-style:italic; font-size:2rem; font-weight:600; margin:0; }
.verdict-fake .verdict-value { color:#cc0000; }
.verdict-real .verdict-value { color:#15803d; }
.signal-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; margin-top:1rem; }
.signal-card { border:0.5px solid #e2e2e2; border-radius:6px; padding:0.85rem 1rem; }
.signal-title { font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; opacity:0.5; margin-bottom:0.3rem; }
.signal-value { font-size:1rem; font-weight:500; }
.tip-box { background:#f8f8f0; border-left:3px solid #d4a800; padding:0.75rem 1rem; font-size:0.82rem; border-radius:0 4px 4px 0; margin-top:1rem; opacity:0.85; }
.mode-badge { display:inline-block; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; padding:0.2rem 0.6rem; border-radius:4px; margin-bottom:0.5rem; }
.demo-file { font-size:0.78rem; font-family:'DM Mono',monospace; }
.folder-fake { color:#cc0000; font-weight:600; }
.folder-real { color:#15803d; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Truthscope")
st.markdown(
    "<p style='font-family:DM Mono;font-size:0.85rem;opacity:0.55;margin-top:-0.5rem;'>"
    "MULTIMODAL FAKE NEWS DETECTOR · BERT + RESNET-18 + MFCC-MLP</p>",
    unsafe_allow_html=True
)
st.divider()

# ── Input section ──────────────────────────────────────────────────────────────
prefill    = st.session_state.pop("sample_text",  "")
pre_video  = st.session_state.pop("demo_video",   None)
pre_audio  = st.session_state.pop("demo_audio",   None)

text_input  = st.text_area(
    "News headline or article text  *(optional)*",
    value=prefill,
    placeholder="Paste a headline from your dataset.json …",
    height=110
)
video_input = st.file_uploader(
    "Upload video  *(optional)*  — MP4 / AVI / MOV",
    type=["mp4", "avi", "mov"]
)
audio_input = st.file_uploader(
    "Upload audio  *(optional)*  — WAV",
    type=["wav"]
)

has_text  = bool(text_input.strip())
has_video = video_input is not None or pre_video is not None
has_audio = audio_input is not None or pre_audio is not None

# Modality badge
_mode_map = {
    (True,  True,  True):  ("🔀 Text + Video + Audio", "#6366f1"),
    (True,  True,  False): ("🔀 Text + Video",          "#6366f1"),
    (True,  False, True):  ("🔀 Text + Audio",          "#8b5cf6"),
    (False, True,  True):  ("🔀 Video + Audio",         "#f59e0b"),
    (True,  False, False): ("📝 Text only",              "#0ea5e9"),
    (False, True,  False): ("🎬 Video only",             "#f59e0b"),
    (False, False, True):  ("🎵 Audio only",             "#ec4899"),
    (False, False, False): (None, None),
}
ml, mc = _mode_map.get((has_text, has_video, has_audio), (None, None))
if ml:
    st.markdown(
        f"<span class='mode-badge' style='background:{mc}22;color:{mc};border:1px solid {mc}55'>{ml}</span>",
        unsafe_allow_html=True
    )

# ── Sample text pills ──────────────────────────────────────────────────────────
with st.expander("Try a sample headline  *(from training data)*"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "<span class='demo-file folder-real'>✅ REAL (label 0)</span>",
            unsafe_allow_html=True
        )
        for s in [
            "JK Rowling sorry for killing off Harry Potter character",
            "Emma Stone Is the Highest-Paid Actress, but Makes Less Than Top 14 Actors",
            "Here's how Jon Snow and Daenerys Targaryen are related in 'Game of Thrones'",
        ]:
            if st.button(s, key=f"r_{s[:20]}"):
                st.session_state["sample_text"] = s
                st.rerun()
    with col2:
        st.markdown(
            "<span class='demo-file folder-fake'>🚨 FAKE (label 1)</span>",
            unsafe_allow_html=True
        )
        for s in [
            "Brad Pitt and Angelina Jolie dismiss reunion reports",
            "Miley Cyrus Pregnancy: The singer wants to have baby with Liam Hemsworth?",
            "Steve Harvey Shuts Down Rumors That He and Wife Marjorie Harvey Are Getting a Divorce",
        ]:
            if st.button(s, key=f"f_{s[:20]}"):
                st.session_state["sample_text"] = s
                st.rerun()
    st.caption(
        "⚠️ Trained on GossipCop + PolitiFact celebrity headlines. "
        "Out-of-distribution topics will give unreliable results."
    )

# ── Demo file picker (video + audio from datasets/ folder) ────────────────────
def _collect_demo_files(base_dir: str, exts):
    """Return list of (filepath, label_str) sorted fake-first."""
    files = []
    for folder_label, lstr in [("fake", "FAKE"), ("real", "REAL")]:
        for ext in exts:
            pattern = os.path.join(base_dir, folder_label, f"*.{ext}")
            for fp in sorted(glob.glob(pattern)):
                files.append((fp, lstr, folder_label))
    return files

VIDEO_DIR = os.path.join("datasets", "videos")
AUDIO_DIR = os.path.join("datasets", "audios")

demo_videos = _collect_demo_files(VIDEO_DIR, ["mp4", "avi", "mov"])
demo_audios = _collect_demo_files(AUDIO_DIR, ["wav"])

if demo_videos or demo_audios:
    with st.expander("🗂️ Use a demo file from `datasets/`  *(shows folder → label)*"):
        if demo_videos:
            st.markdown("**Videos**")
            for fp, lstr, folder in demo_videos:
                fname   = os.path.basename(fp)
                col_cls = "folder-fake" if folder == "fake" else "folder-real"
                icon    = "🚨" if folder == "fake" else "✅"
                label_html = (
                    f"<span class='demo-file'>{icon} "
                    f"<code>{fname}</code> "
                    f"— folder: <span class='{col_cls}'>{folder}/</span> "
                    f"→ label <b>{lstr}</b></span>"
                )
                btn_col, info_col = st.columns([1, 3])
                with btn_col:
                    if st.button("Use", key=f"dv_{fp}"):
                        st.session_state["demo_video"] = fp
                        st.rerun()
                with info_col:
                    st.markdown(label_html, unsafe_allow_html=True)

        if demo_audios:
            st.markdown("**Audio files**")
            for fp, lstr, folder in demo_audios:
                fname   = os.path.basename(fp)
                col_cls = "folder-fake" if folder == "fake" else "folder-real"
                icon    = "🚨" if folder == "fake" else "✅"
                label_html = (
                    f"<span class='demo-file'>{icon} "
                    f"<code>{fname}</code> "
                    f"— folder: <span class='{col_cls}'>{folder}/</span> "
                    f"→ label <b>{lstr}</b></span>"
                )
                btn_col, info_col = st.columns([1, 3])
                with btn_col:
                    if st.button("Use", key=f"da_{fp}"):
                        st.session_state["demo_audio"] = fp
                        st.rerun()
                with info_col:
                    st.markdown(label_html, unsafe_allow_html=True)

        st.caption(
            "Files under `datasets/videos/fake/` are labelled FAKE (1); "
            "`datasets/videos/real/` → REAL (0). Same convention for audio."
        )

st.markdown(
    '<div class="tip-box">💡 <b>At least one input required.</b> '
    'Video → middle frame extracted. Audio → mean MFCCs (40-dim). '
    'Missing modalities fall back to zero vectors automatically.</div>',
    unsafe_allow_html=True
)
st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔍 Analyse", type="primary", use_container_width=True):
    if not has_text and not has_video and not has_audio:
        st.warning("Please provide at least a headline, video, or audio file before analysing.")
    else:
        with st.spinner("Running inference…"):

            # ── resolve video path ───────────────────────────────────────────
            video_path = None
            tmp_video  = None
            if video_input is not None:
                suffix    = os.path.splitext(video_input.name)[-1]
                tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_video.write(video_input.read()); tmp_video.flush()
                video_path = tmp_video.name
            elif pre_video is not None:
                video_path = pre_video

            # ── resolve audio path ───────────────────────────────────────────
            audio_path = None
            tmp_audio  = None
            if audio_input is not None:
                tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_audio.write(audio_input.read()); tmp_audio.flush()
                audio_path = tmp_audio.name
            elif pre_audio is not None:
                audio_path = pre_audio

            try:
                result = predict(
                    text       = text_input.strip() if has_text else "",
                    video_path = video_path,
                    audio_path = audio_path,
                )
            except Exception as e:
                st.error(f"Inference error: {e}")
                result = None
            finally:
                for tmp in [tmp_video, tmp_audio]:
                    if tmp:
                        tmp.close()
                        try: os.unlink(tmp.name)
                        except Exception: pass

        if result:
            label      = result["label"]
            fake_prob  = result["fake_prob"]
            real_prob  = result["real_prob"]
            confidence = result["confidence"]
            modality   = result.get("modality", "—")

            vcls  = "verdict-fake" if label == "FAKE" else "verdict-real"
            vicon = "🚨" if label == "FAKE" else "✅"

            st.markdown(f"""
            <div class="{vcls}">
                <div class="verdict-label">Verdict</div>
                <div class="verdict-value">{vicon} {label}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**REAL probability**")
                st.progress(real_prob, text=f"{real_prob:.1%}")
            with c2:
                st.markdown("**FAKE probability**")
                st.progress(fake_prob, text=f"{fake_prob:.1%}")

            tier  = "HIGH" if confidence >= 0.80 else "MEDIUM" if confidence >= 0.60 else "LOW"
            tcol  = {"HIGH": "#15803d", "MEDIUM": "#d97706", "LOW": "#cc0000"}[tier]
            tsig  = ("N/A" if not has_text
                     else "Suspicious" if fake_prob > 0.6 else "Credible")
            vsig  = ("N/A" if not has_video
                     else "Suspicious" if fake_prob > 0.6 else "Credible")
            asig  = ("N/A" if not has_audio
                     else "Suspicious" if fake_prob > 0.6 else "Credible")

            st.markdown(f"""
            <div class="signal-grid">
                <div class="signal-card">
                    <div class="signal-title">Confidence</div>
                    <div class="signal-value" style="color:{tcol}">{confidence:.1%} · {tier}</div>
                </div>
                <div class="signal-card">
                    <div class="signal-title">Modality</div>
                    <div class="signal-value">{modality}</div>
                </div>
                <div class="signal-card">
                    <div class="signal-title">Text signal</div>
                    <div class="signal-value">{tsig}</div>
                </div>
                <div class="signal-card">
                    <div class="signal-title">Video signal</div>
                    <div class="signal-value">{vsig}</div>
                </div>
                <div class="signal-card">
                    <div class="signal-title">Audio signal</div>
                    <div class="signal-value">{asig}</div>
                </div>
                <div class="signal-card">
                    <div class="signal-title">Model</div>
                    <div class="signal-value">BERT + ResNet-18 + MFCC</div>
                </div>
            </div>""", unsafe_allow_html=True)

            with st.expander("Raw scores"):
                st.json(result)