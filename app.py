MODEL_URL = "https://drive.google.com/file/d/1MTETSfmvMlTgX55iT7kGbOFBU2rF9B3R/view"
MODEL_PATH = "brain_tumor_model.keras"
MIN_SIZE_MB = 50  # expected ~117 MB

# ---- Clean up any bad cached file ----
if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if size_mb < MIN_SIZE_MB:
        os.remove(MODEL_PATH)

# ---- Download model safely ----
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        gdown.download(
            MODEL_URL,
            MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

# ---- Validate file before loading ----
size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
if size_mb < MIN_SIZE_MB:
    st.error(
        f"Model download failed (only {size_mb:.1f} MB). "
        "Please check Google Drive permissions."
    )
    st.stop()

# ---- Load model (inference only) ----
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH, compile=False)

model = load_trained_model()
