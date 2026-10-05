import os
import sys
import json
import cv2
import numpy as np
import gradio as gr
from PIL import Image

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import run_pipeline
import forensic_analysis

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Đang khởi tạo mô hình Giám định AI trên thiết bị: {device}...")
detector = run_pipeline.HybridForensicDetector(device=device)
print(f"[*] Khởi tạo thành công trên {device}!")

def analyze_uploaded_image(input_image):
    if input_image is None:
        return (
            "<div style='color: gray; font-size: 16px;'>Vui lòng tải lên một bức ảnh để bắt đầu giám định.</div>",
            None,
            None
        )
    
    temp_dir = "out"
    os.makedirs(temp_dir, exist_ok=True)
    temp_input_path = os.path.join(temp_dir, "web_temp_input.png")
    
    if isinstance(input_image, dict) and "path" in input_image:
        input_image = input_image["path"]

    if isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(input_image)
    elif isinstance(input_image, str):
        if not os.path.exists(input_image):
            return (
                "<div style='color: #ef4444; font-size: 16px;'>⚠️ Tệp ảnh tạm thời không tìm thấy do phiên hết hạn. Vui lòng chọn lại ảnh và bấm Giám định.</div>",
                None,
                None
            )
        pil_img = Image.open(input_image).convert('RGB')
    else:
        pil_img = Image.open(input_image).convert('RGB')
        
    pil_img.save(temp_input_path)
    
    # 1. Chạy phân tích tổng hợp (Pipeline)
    result_data = detector.analyze_image(temp_input_path, out_dir=temp_dir)
    
    # 2. Sinh đồ thị phân tích chi tiết 6 Panel
    forensic_plot_path = os.path.join(temp_dir, "web_forensic_plot.png")
    forensic_analysis.run_forensic_pipeline(temp_input_path, forensic_plot_path)
    
    status = result_data["status"]
    conf = result_data["confidence_percent"]
    logit = result_data["bfree_logit"]
    local_logit = result_data.get("max_local_logit", logit)
    srm_ratio = result_data["srm_high_energy_ratio"] * 100
    has_wm = result_data.get("ai_watermark_detected", result_data.get("gemini_watermark_detected", False))
    
    if "REAL" in status:
        accent = "#10b981"
        bg_card = "rgba(16, 185, 129, 0.12)"
        border_color = "#10b981"
        badge_text = "AUTHENTIC REAL — ẢNH THẬT TỰ NHIÊN"
        icon = "✓"
    elif "LOCAL" in status:
        accent = "#f59e0b"
        bg_card = "rgba(245, 158, 11, 0.12)"
        border_color = "#f59e0b"
        badge_text = "LOCAL INPAINTING — ẢNH THẬT BỊ CHỈNH SỬA AI CỤC BỘ"
        icon = "⚠"
    else:
        accent = "#ef4444"
        bg_card = "rgba(239, 68, 68, 0.12)"
        border_color = "#ef4444"
        badge_text = "AI-GENERATED — ẢNH DO AI TẠO TOÀN PHẦN"
        icon = "✕"

    verdict_html = f"""
    <div style="
        background: {bg_card};
        border: 2px solid {border_color};
        border-radius: 14px;
        padding: 22px 26px;
        font-family: 'Be Vietnam Pro', 'Segoe UI', sans-serif;
        box-shadow: 0 0 30px {border_color}33;
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        ">
            <div style="
                width: 36px; height: 36px;
                background: {border_color};
                border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                font-size: 18px; font-weight: 900; color: white;
                flex-shrink: 0;
            ">{icon}</div>
            <div style="
                font-size: 17px;
                font-weight: 800;
                color: {accent};
                letter-spacing: 0.5px;
                line-height: 1.3;
            ">{badge_text}</div>
        </div>
        <div style="
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        ">
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 14px;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Độ tin cậy</div>
                <div style="font-size: 20px; font-weight: 800; color: #f1f5f9;">{conf:.1f}%</div>
            </div>
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 14px;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Điểm Logit B-Free</div>
                <div style="font-size: 18px; font-weight: 800; color: #f1f5f9;">{logit:+.3f} <span style="font-size: 12px; font-weight: 500; color: #94a3b8;">(Max Crop: {local_logit:+.2f})</span></div>
            </div>
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 14px;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Bất thường nhiễu SRM</div>
                <div style="font-size: 20px; font-weight: 800; color: #f1f5f9;">{srm_ratio:.2f}%</div>
            </div>
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 14px;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Dấu hiệu bản quyền AI</div>
                <div style="font-size: 20px; font-weight: 800; color: {'#ef4444' if has_wm else '#10b981'};">{'CÓ' if has_wm else 'KHÔNG'}</div>
            </div>
        </div>
    </div>
    """
    
    plot_result = Image.open(forensic_plot_path)
    json_result = json.dumps(result_data, indent=4, ensure_ascii=False)
    
    return verdict_html, plot_result, json_result

custom_css = """
* {
    font-family: 'Be Vietnam Pro', 'Segoe UI', sans-serif !important;
}

html, body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    max-width: 98% !important;
    width: 98% !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: transparent !important;
}

/* Header */
.gradio-container h1 {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 26px !important;
    font-weight: 900 !important;
    letter-spacing: 1px !important;
    color: #ffffff !important;
    text-align: center !important;
    padding: 28px 0 18px 0 !important;
    margin: 0 !important;
    border: none !important;
    text-shadow: 0 0 40px rgba(99, 102, 241, 0.6) !important;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

h3, .gradio-container h3 {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

/* Panel blocks */
.block, .gr-block, .gr-box {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
}

/* Upload area */
.upload-container, [data-testid="image"] {
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* Buttons */
button[variant="primary"], .primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 10px !important;
    border: none !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
}

button[variant="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(99, 102, 241, 0.6) !important;
}

button[variant="secondary"], .secondary {
    background: rgba(255,255,255,0.08) !important;
    color: #cbd5e1 !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}

/* Labels */
label span, .label-wrap span {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* Accordion */
.accordion {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

/* Code block */
.code-wrap, pre {
    background: #0f172a !important;
    color: #7dd3fc !important;
    font-size: 13px !important;
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
"""

with gr.Blocks(
    title="He Thong Giam Dinh Anh",
    css=custom_css,
    theme=gr.themes.Base(
        primary_hue="violet",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Be Vietnam Pro"),
    ).set(
        body_background_fill="transparent",
        body_background_fill_dark="transparent",
        block_background_fill="rgba(255,255,255,0.04)",
        block_background_fill_dark="rgba(255,255,255,0.04)",
        block_border_color="rgba(255,255,255,0.08)",
        block_border_color_dark="rgba(255,255,255,0.08)",
        block_label_text_color="#94a3b8",
        block_label_text_color_dark="#94a3b8",
        input_background_fill="rgba(255,255,255,0.06)",
        input_background_fill_dark="rgba(255,255,255,0.06)",
        input_border_color="rgba(255,255,255,0.1)",
        input_border_color_dark="rgba(255,255,255,0.1)",
        button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
        button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
        button_primary_text_color="white",
        button_secondary_background_fill="rgba(255,255,255,0.08)",
        button_secondary_background_fill_dark="rgba(255,255,255,0.08)",
        button_secondary_text_color="#cbd5e1",
        button_secondary_text_color_dark="#cbd5e1",
    )
) as demo:
    # Inject Google Fonts via HTML (CSS @import blocked by Gradio sandbox)
    gr.HTML("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
        <style>*, body, .gradio-container { font-family: 'Be Vietnam Pro', sans-serif !important; }</style>
    """)
    gr.Markdown("# HỆ THỐNG GIÁM ĐỊNH ẢNH SỐ VÀ PHÁT HIỆN AI")

    with gr.Row(equal_height=False):
        # LEFT: Upload
        with gr.Column(scale=5, min_width=420):
            gr.Markdown("### Tải ảnh lên")
            input_img = gr.Image(type="numpy", label="Chọn ảnh từ máy tính", height=420)
            with gr.Row():
                btn_clear = gr.ClearButton([input_img], value="Xóa ảnh", size="lg")
                btn_submit = gr.Button("Bắt đầu Giám định", variant="primary", size="lg")

        # RIGHT: Results
        with gr.Column(scale=5, min_width=420):
            gr.Markdown("### Kết quả Giám định")
            output_verdict = gr.HTML("""
                <div style="
                    background: rgba(255,255,255,0.04);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 12px;
                    padding: 20px 24px;
                    color: #64748b;
                    font-family: 'Be Vietnam Pro', sans-serif;
                    font-size: 15px;
                ">
                    Chưa có kết quả giám định. Hãy chọn ảnh và bấm 'Bắt đầu Giám định'.
                </div>
            """)
            output_plot = gr.Image(label="Đồ thị Giám định Đối chiếu Đa tầng (Ảnh gốc vs Zoom)", interactive=False)
            with gr.Accordion("Dữ liệu kỹ thuật JSON", open=False):
                output_json = gr.Code(label="JSON", language="json")

    btn_submit.click(
        fn=analyze_uploaded_image,
        inputs=[input_img],
        outputs=[output_verdict, output_plot, output_json]
    )

if __name__ == '__main__':
    print("[*] Dang khoi chay Web Server...", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

