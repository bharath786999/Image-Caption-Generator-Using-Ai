#!/usr/bin/env python3
"""
image_caption_generator.py

Single-file image caption generator.

Usage (CLI):
    python image_caption_generator.py --image /path/to/photo.jpg

Run tiny web UI (Flask):
    python image_caption_generator.py --web
    Then open http://127.0.0.1:5000 in your browser.

Use Hugging Face Inference API instead of local model:
    - Option A: set environment variable HF_API_TOKEN (recommended for iPad/Colab if you don't want to download model)
    - Option B: pass --use-api

Required packages:
    pip install transformers torch pillow flask requests
If you only want API mode (no model download), you still need requests and pillow:
    pip install requests pillow flask

Model used for local mode:
    "Salesforce/blip-image-captioning-base"

Notes:
 - Local mode requires enough RAM and a working PyTorch install.
 - API mode uses Hugging Face inference endpoint and requires an API token.
"""

import os
import sys
import argparse
import io

from PIL import Image

# Optional imports (we'll check availability)
_have_transformers = False
_have_torch = False
_have_flask = False
_have_requests = False

try:
    import transformers
    _have_transformers = True
except Exception:
    _have_transformers = False

try:
    import torch
    _have_torch = True
except Exception:
    _have_torch = False

try:
    from flask import Flask, request, render_template_string, send_from_directory
    _have_flask = True
except Exception:
    _have_flask = False

try:
    import requests
    _have_requests = True
except Exception:
    _have_requests = False

# ---------------------------
# Helper: pip install hint
# ---------------------------
def pip_install_hint(packages):
    print("\nIf you need to install packages, run:")
    print("    pip install " + " ".join(packages))
    print("On Colab, prefix with ! in a cell, e.g.:")
    print("    !pip install " + " ".join(packages))
    print()

# ---------------------------
# Hugging Face Inference API
# ---------------------------
def caption_via_hf_api(image_path_or_fileobj, hf_token=None, model="Salesforce/blip-image-captioning-base", max_length=40):
    """
    image_path_or_fileobj: path string or file-like object (binary)
    hf_token: string. If None, will read HF_API_TOKEN env var.
    """
    if not _have_requests:
        raise RuntimeError("requests library not installed. pip install requests")
    token = hf_token or os.environ.get("HF_API_TOKEN")
    if not token:
        raise RuntimeError("Hugging Face API token not provided. Set HF_API_TOKEN env var or pass hf_token.")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    # handle both path and file-like object
    if isinstance(image_path_or_fileobj, str):
        f = open(image_path_or_fileobj, "rb")
        close_after = True
    else:
        f = image_path_or_fileobj
        close_after = False
    files = {"inputs": f}
    params = {"options": {"wait_for_model": True}, "parameters": {"max_new_tokens": max_length}}
    # The inference API accepts form posts (some models) — handle basic usage:
    resp = requests.post(url, headers=headers, files=files, params=None, data=None, timeout=120)
    if close_after:
        f.close()
    if resp.status_code != 200:
        raise RuntimeError(f"Hugging Face API error {resp.status_code}: {resp.text}")
    # response likely a list of dicts or dict with generated_text - handle both
    j = resp.json()
    # common format: [{'generated_text': 'A dog ...'}]
    if isinstance(j, list) and len(j) > 0 and isinstance(j[0], dict):
        # try many keys
        for key in ("generated_text", "caption", "text"):
            if key in j[0]:
                return j[0][key].strip()
        # fallback: join values
        return " ".join(str(v) for v in j[0].values()).strip()
    elif isinstance(j, dict):
        # maybe {'generated_text': '...'}
        for key in ("generated_text", "caption", "text"):
            if key in j:
                return j[key].strip()
        return str(j).strip()
    else:
        return str(j)

# ---------------------------
# Local pipeline (transformers)
# ---------------------------
_caption_pipeline = None
def ensure_local_pipeline(model_name="Salesforce/blip-image-captioning-base"):
    global _caption_pipeline
    if _caption_pipeline is not None:
        return _caption_pipeline
    if not _have_transformers:
        raise RuntimeError("transformers not installed. pip install transformers")
    if not _have_torch:
        raise RuntimeError("torch not installed. pip install torch")
    # create pipeline
    try:
        from transformers import pipeline
        # set device to 0 if CUDA available, else cpu
        device = 0 if torch.cuda.is_available() else -1
        _caption_pipeline = pipeline("image-to-text", model=model_name, device=device)
        return _caption_pipeline
    except Exception as e:
        raise RuntimeError("Failed to create pipeline: " + str(e))

def generate_caption_local(image_path_or_pil, max_length=40):
    """
    Accepts path string or PIL.Image
    """
    pipe = ensure_local_pipeline()
    # the pipeline accepts file path or PIL image
    inputs = image_path_or_pil
    try:
        out = pipe(inputs, max_length=max_length, truncation=True)
        if isinstance(out, list) and len(out) > 0:
            # usually [{'generated_text': '...'}]
            txt = out[0].get("generated_text") if isinstance(out[0], dict) else str(out[0])
            return (txt or "").strip()
        # fallback
        return str(out)
    except Exception as e:
        raise RuntimeError("Model inference failed: " + str(e))

# ---------------------------
# Unified wrapper
# ---------------------------
def generate_caption(image_path, use_api=False, hf_token=None, model_name="Salesforce/blip-image-captioning-base", max_length=40):
    """
    image_path: path to image file
    use_api: if True, use Hugging Face Inference API
    hf_token: optional token
    """
    if use_api:
        return caption_via_hf_api(image_path, hf_token=hf_token, model=model_name, max_length=max_length)
    else:
        return generate_caption_local(image_path, max_length=max_length)

# ---------------------------
# CLI and Flask web UI
# ---------------------------
HTML_TEMPLATE = """
<!doctype html>
<title>Image Caption Generator</title>
<h1>Image Caption Generator</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*">
  <br><br>
  <label><input type="checkbox" name="use_api" {{ 'checked' if use_api else '' }}> Use Hugging Face API (requires HF_API_TOKEN env var)</label>
  <br><br>
  <input type=submit value='Upload & Caption'>
</form>
{% if filename %}
  <h2>Uploaded</h2>
  <img src="/uploads/{{ filename }}" style="max-width:400px;">
  <h3>Caption</h3>
  <div style="background:#f7f7f7;padding:10px;border-radius:6px;">{{ caption }}</div>
{% endif %}
<hr>
<p>Notes: For local mode you must have <code>transformers</code> and <code>torch</code> installed.
For API mode set environment variable <code>HF_API_TOKEN</code>.</p>
"""

def run_flask_server(model_name="Salesforce/blip-image-captioning-base"):
    if not _have_flask:
        raise RuntimeError("Flask not installed. pip install flask")
    from flask import Flask, request, render_template_string, send_from_directory
    app = Flask(_name_)
    upload_folder = os.path.join(os.path.dirname(_file_), "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    @app.route("/", methods=["GET", "POST"])
    def index():
        caption = None
        filename = None
        use_api = False
        if request.method == "POST":
            f = request.files.get("file")
            use_api = "use_api" in request.form
            if f:
                filename = f.filename
                save_path = os.path.join(upload_folder, filename)
                f.save(save_path)
                try:
                    caption = generate_caption(save_path, use_api=use_api)
                except Exception as e:
                    caption = "ERROR: " + str(e)
        return render_template_string(HTML_TEMPLATE, caption=caption, filename=filename, use_api=use_api)

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(upload_folder, filename)

    print("Starting Flask web UI on http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Image caption generator (local model or HF inference API).")
    parser.add_argument("--image", "-i", help="Path to image file to caption.")
    parser.add_argument("--web", action="store_true", help="Run small Flask web UI.")
    parser.add_argument("--use-api", action="store_true", help="Use Hugging Face Inference API instead of local model.")
    parser.add_argument("--hf-token", help="Hugging Face API token (optional). If not provided, reads HF_API_TOKEN env var.")
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base", help="Model name to use (local or API).")
    parser.add_argument("--max-length", type=int, default=40, help="Max tokens / length for caption generation.")
    args = parser.parse_args()

    # If user asked for web UI
    if args.web:
        print("Web UI selected.")
        # If user wants API mode, Flask form will have checkbox; but we can let it run local by default.
        if not _have_flask:
            print("Flask not available. Install with: pip install flask")
            pip_install_hint(["flask", "pillow", "requests", "transformers", "torch"])
            sys.exit(1)
        # If user likely wants local model but transformers not installed, tell them
        if not args.use_api and (not _have_transformers or not _have_torch):
            print("Warning: transformers/torch not installed. Running web UI in local model mode will fail until installed.")
            print("You can use the checkbox 'Use Hugging Face API' in the UI if you have HF_API_TOKEN.")
            pip_install_hint(["transformers", "torch", "pillow"])
        run_flask_server(model_name=args.model)
        return

    # CLI image caption
    if args.image:
        img_path = args.image
        if not os.path.isfile(img_path):
            print("Image file not found:", img_path)
            sys.exit(1)
        # Decide mode
        use_api = args.use_api or (os.environ.get("HF_API_TOKEN") is not None)
        if use_api and not _have_requests:
            print("requests is required for API mode. Install with: pip install requests")
            pip_install_hint(["requests"])
            sys.exit(1)
        if (not use_api) and (not _have_transformers or not _have_torch):
            print("Local mode requested but 'transformers' and/or 'torch' are not installed.")
            pip_install_hint(["transformers", "torch", "pillow"])
            print("To avoid downloading a model locally, rerun with --use-api and set HF_API_TOKEN env variable.")
            sys.exit(1)

        print("Generating caption (model=%s) ..." % args.model)
        try:
            caption = generate_caption(img_path, use_api=use_api, hf_token=args.hf_token, model_name=args.model, max_length=args.max_length)
            print("Caption:")
            print(caption)
        except Exception as e:
            print("Failed to generate caption:", e)
            print("If you're on an iPad, consider using the Hugging Face Inference API mode (--use-api) with HF_API_TOKEN set.")
            sys.exit(1)
        return

    # No args: show help
    parser.print_help()

if_name_ == "_main_":
    main()