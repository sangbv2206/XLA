import os
import modal

# Modal App definition
app = modal.App("bfree-forensics")

# Define Linux Cloud Container on Modal
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "wget", "unzip")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "gradio",                         # Latest Gradio 5 (upload_progress fixed by max_containers=1)
        "python-multipart",       # Required for FastAPI to parse multipart file uploads
        "timm>=1.0.12",
        "transformers>=4.43.4",
        "scikit-learn",
        "opencv-python-headless",
        "scipy",
        "pillow",
        "matplotlib",
        "pyyaml",
        "fastapi",
        extra_index_url="https://download.pytorch.org/whl/cpu"
    )
    # Download weights directly inside container layer (10Gbps network)
    .run_commands(
        "mkdir -p /root/app/weights && "
        "wget -q https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip -O /root/app/weights/BFREE_dino2reg4.zip && "
        "unzip -q -o /root/app/weights/BFREE_dino2reg4.zip -d /root/app/weights/ && "
        "rm -f /root/app/weights/BFREE_dino2reg4.zip"
    )
    # Mount local python files to container
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=["weights", "*.zip", "__pycache__", "out", "*.ipynb", ".git", "build_colab_notebook.py"]
    )
)

# Persistent volume for Gradio uploads so files are never lost on container restart
upload_vol = modal.Volume.from_name("bfree-upload-cache", create_if_missing=True)

# Serverless ASGI endpoint
@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=600,
    scaledown_window=300, # Giữ container sống 5 phút, không bị tắt giữa lúc upload và click
    max_containers=1,     # Đảm bảo mọi request và session tập trung 1 container
    volumes={"/tmp/gradio": upload_vol}
)
@modal.asgi_app()
def web():
    import sys
    os.chdir("/root/app")
    if "/root/app" not in sys.path:
        sys.path.insert(0, "/root/app")
    
    from fastapi import FastAPI
    import gradio as gr
    import app as web_ui
    
    server = FastAPI()
    return gr.mount_gradio_app(
        server,
        web_ui.demo.queue(),  # .queue() needed for upload handling and callbacks
        path="/"
    )
