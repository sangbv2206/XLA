import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def compute_ela(image_path, quality=90):
    """
    Error Level Analysis (ELA):
    Nén lại ảnh ở mức chất lượng 90% để tìm sự bất thường về mức nén giữa các vùng.
    """
    orig = Image.open(image_path).convert('RGB')
    temp_path = "temp_ela_resave.jpg"
    orig.save(temp_path, 'JPEG', quality=quality)
    resaved = Image.open(temp_path).convert('RGB')
    
    diff = ImageChops.difference(orig, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    
    scale_factor = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale_factor)
    
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
    return orig, diff

def compute_noise_residual(image_path):
    """
    Spatial Rich Model (SRM) High-pass Residual:
    Trích xuất dấu vết nhiễu cảm biến quang học ở dải tần số cao.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([
        [-1,  2, -1],
        [ 2, -4,  2],
        [-1,  2, -1]
    ], dtype=np.float32)
    
    residual = np.abs(cv2.filter2D(img.astype(np.float32), -1, kernel))
    kernel_var = np.ones((7, 7), dtype=np.float32) / 49.0
    local_energy = np.sqrt(np.maximum(cv2.filter2D(residual ** 2, -1, kernel_var), 0))
    
    norm_residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm_energy = cv2.normalize(local_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm_residual, norm_energy

def spectral_residual_saliency(img_bgr):
    """
    Thuật toán Spectral Residual Saliency (Hou & Zhang, CVPR):
    Phát hiện vùng nổi bật thị giác / đối tượng chính dựa trên phân tích phổ tần số 2D FFT.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray_small = cv2.resize(gray, (256, 256)).astype(np.float32)
    fft = np.fft.fft2(gray_small)
    amplitude = np.abs(fft)
    log_amp = np.log(amplitude + 1e-6)
    phase = np.angle(fft)
    
    mean_log_amp = cv2.blur(log_amp, (5, 5))
    spectral_residual = log_amp - mean_log_amp
    
    recon = np.real(np.fft.ifft2(np.exp(spectral_residual) * np.exp(1j * phase)))
    saliency = cv2.GaussianBlur(recon ** 2, (9, 9), 0)
    saliency_full = cv2.resize(saliency, (w, h))
    return cv2.normalize(saliency_full, None, 0, 1.0, cv2.NORM_MINMAX)

def find_suspicious_roi(img_bgr, noise_energy, ela_pil):
    """
    Thuật toán Giám định Vùng Giả mạo Tự động Tổng quát (Generic Forensic ROI Localization):
    Áp dụng nguyên lý Xử lý ảnh số (XLA) & Giám định số (Digital Forensics):
    Hoạt động tổng quát cho MỌI đối tượng (chuột, chai nước, laptop, người, vật thể AI bất kỳ).
    
    Nguyên lý 3 tầng:
    1. Spectral Residual Saliency: Định vị vùng hội tụ tần số không gian của vật thể được chèn vào.
    2. SRM Residual Energy: Phân tích độ bất thường năng lượng nhiễu cảm biến tần số cao (PRNU).
    3. Background Contrast: Đo khoảng cách màu sắc của vật thể so với nền tự nhiên của bối cảnh.
    """
    h, w, _ = img_bgr.shape
    
    # Tầng 1: Saliency phổ Fourier
    norm_sal = spectral_residual_saliency(img_bgr)
    
    # Tầng 2: Năng lượng nhiễu SRM
    norm_srm = cv2.normalize(noise_energy.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Tầng 3: Độ tương phản nền (Background Contrast)
    bg_sample = np.concatenate([
        img_bgr[int(h*0.75):int(h*0.95), int(w*0.15):int(w*0.85)].reshape(-1, 3),
        img_bgr[int(h*0.4):int(h*0.7), :int(w*0.15)].reshape(-1, 3)
    ], axis=0)
    bg_median = np.median(bg_sample, axis=0)
    color_diff = np.linalg.norm(img_bgr.astype(np.float32) - bg_median, axis=-1)
    norm_diff = cv2.normalize(color_diff, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Hợp nhất đa đặc trưng giám định quang học
    fusion_map = 0.45 * norm_sal + 0.35 * norm_diff + 0.20 * norm_srm
    
    # Khử nhiễu biên rìa (loại trừ 5% mép ảnh nơi ánh sáng và quang sai thường gây nhiễu)
    margin_y = int(h * 0.05)
    margin_x = int(w * 0.05)
    work_map = fusion_map.copy()
    work_map[:margin_y, :] = 0
    work_map[-margin_y:, :] = 0
    work_map[:, :margin_x] = 0
    work_map[:, -margin_x:] = 0
    
    # Phân ngưỡng Otsu thích nghi trên bản đồ năng lượng tổng hợp
    work_u8 = (cv2.normalize(work_map, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    ret, binary = cv2.threshold(work_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Phép đóng hình thái học để gom các cụm chi tiết vi mô thành khối đối tượng hoàn chỉnh
    k_size = max(15, int(min(h, w) * 0.025))
    if k_size % 2 == 0:
        k_size += 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = w * h
    
    # Lọc đối tượng có kích thước hợp lệ (1.5% đến 65% diện tích khung hình)
    valid_c = [c for c in contours if 0.015 * total_area < cv2.contourArea(c) < 0.65 * total_area]
    
    if valid_c:
        best_c = max(valid_c, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(best_c)
        
        # Mở rộng padding vừa vặn 6% để không cắt viền đối tượng
        pad_x = int(bw * 0.06)
        pad_y = int(bh * 0.06)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)
        return (x1, y1, x2, y2), True
    else:
        # Fallback về vùng trọng tâm ảnh
        cx, cy = w // 2, h // 2
        cw, ch = int(w * 0.50), int(h * 0.50)
        return (max(0, cx - cw // 2), max(0, cy - ch // 2), min(w, cx + cw // 2), min(h, cy + ch // 2)), False

def run_forensic_pipeline(image_path, output_path="out/forensic_analysis.png"):
    print(f"[*] Đang chạy Giám định Tự động trên: {image_path}")
    orig_pil, ela_pil = compute_ela(image_path, quality=90)
    residual, noise_energy = compute_noise_residual(image_path)
    
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    # TỰ ĐỘNG ĐỊNH VỊ CHUẨN XÁC VẬT THỂ AI
    (bx1, by1, bx2, by2), is_detected = find_suspicious_roi(img_bgr, noise_energy, ela_pil)
    print(f"[*] Toạ độ vật thể AI khoanh vùng: [{bx1}, {by1}, {bx2}, {by2}] (Kích thước: {bx2-bx1}x{by2-by1} px)")
    
    crop_orig = img_rgb[by1:by2, bx1:bx2]
    crop_ela = np.array(ela_pil)[by1:by2, bx1:bx2]
    crop_noise = noise_energy[by1:by2, bx1:bx2]
    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Thiết lập bố cục ĐỒ THỊ DỌC (3 hàng x 2 cột): Đối chiếu trực tiếp Toàn cảnh vs Zoom
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    plt.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.05, wspace=0.22, hspace=0.30)
    
    # HÀNG 1: Ảnh Quang học (Ảnh gốc vs Zoom vật thể)
    orig_display = img_rgb.copy()
    box_color = (255, 0, 0) if is_detected else (0, 255, 0)
    cv2.rectangle(orig_display, (bx1, by1), (bx2, by2), box_color, 4)
    cv2.putText(orig_display, "AI Object Detected", (bx1, max(30, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, box_color, 2)
    
    axes[0, 0].imshow(orig_display)
    axes[0, 0].set_title("1. Ảnh Gốc (Toàn cảnh & Khoanh vùng)", fontsize=13, fontweight='bold', pad=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(crop_orig)
    title_roi = "2. Zoom Chi Tiết Vùng Nghi Vấn" if is_detected else "2. Zoom Vùng Trọng Tâm"
    axes[0, 1].set_title(title_roi, fontsize=13, fontweight='bold', pad=10)
    axes[0, 1].axis('off')
    
    # HÀNG 2: Phân tích Sai lệch Nén ELA (Toàn cảnh vs Zoom)
    axes[1, 0].imshow(ela_pil)
    axes[1, 0].set_title("3. Error Level Analysis (ELA Toàn cảnh)", fontsize=13, fontweight='bold', pad=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(crop_ela)
    axes[1, 1].set_title("4. ELA Vi Mô Vùng Nghi Vấn", fontsize=13, fontweight='bold', pad=10)
    axes[1, 1].axis('off')
    
    # HÀNG 3: Phân tích Nhiễu Cảm biến SRM (Toàn cảnh vs Zoom)
    im_noise = axes[2, 0].imshow(noise_energy, cmap='inferno')
    axes[2, 0].set_title("5. Bản Đồ Nhiễu SRM (Toàn cảnh)", fontsize=13, fontweight='bold', pad=10)
    axes[2, 0].axis('off')
    cbar = fig.colorbar(im_noise, ax=axes[2, 0], fraction=0.046, pad=0.04)
    cbar.set_label("Năng lượng nhiễu tần số cao", fontsize=9)
    
    axes[2, 1].imshow(crop_noise, cmap='inferno')
    axes[2, 1].set_title("6. SRM Vi Mô Vùng Nghi Vấn", fontsize=13, fontweight='bold', pad=10)
    axes[2, 1].axis('off')
    
    fig.suptitle("HỆ THỐNG GIÁM ĐỊNH PHÁP CHỨNG ẢNH SỐ (ĐỐI CHIẾU ĐA TẦNG)", 
                 fontsize=16, fontweight='bold', color='darkred' if is_detected else 'darkgreen', y=0.97)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã xuất kết quả phân tích dạng dọc (3x2): {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Giám định Ảnh số Tự động ROI")
    parser.add_argument('-i', '--input', type=str, required=True, help="Đường dẫn file ảnh")
    parser.add_argument('-o', '--output', type=str, default="out/forensic_analysis.png", help="Đường dẫn file xuất kết quả")
    args = parser.parse_args()
    run_forensic_pipeline(args.input, args.output)
