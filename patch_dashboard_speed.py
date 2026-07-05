from pathlib import Path

APP_PATH = Path("pipeline/dashboard/app.py")
text = APP_PATH.read_text(encoding="utf-8")

# Make cloud live mode less aggressive
text = text.replace("UI_REFRESH_SECS = 5", "UI_REFRESH_SECS = 30")
text = text.replace("STATUS_CACHE_SECS = 15", "STATUS_CACHE_SECS = 60")

# Do not auto-refresh by default
text = text.replace(
    'auto_refresh = st.sidebar.toggle("Auto-refresh every 5s", value=(mode == "Live (read pipeline)"))',
    'auto_refresh = st.sidebar.toggle("Auto-refresh every 30s", value=False)'
)

# If current compatible version reads full table using dt.to_pandas(),
# switch to reading only dashboard columns.
old = '''        try:
            df = dt.to_pandas()
        except Exception as read_exc:
            return False, f"Gold table unreadable: {read_exc}", None
'''

new = '''        try:
            desired_cols = sorted(required_cols | optional_cols)
            df = dt.to_pandas(columns=desired_cols)
        except Exception:
            try:
                df = dt.to_pandas(columns=sorted(required_cols))
            except Exception as read_exc:
                return False, f"Gold table unreadable: {read_exc}", None
'''

if old in text:
    text = text.replace(old, new, 1)
else:
    print("WARNING: Exact dt.to_pandas() block not found. Refresh settings were still patched.")

APP_PATH.write_text(text, encoding="utf-8")

print("Dashboard speed patch applied.")
print("Refresh changed from 5s to 30s.")
print("Auto-refresh default changed to OFF.")
print("Delta read changed to selected columns when possible.")