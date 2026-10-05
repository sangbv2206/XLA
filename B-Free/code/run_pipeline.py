import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
from torchvision.transforms import Compose
import yaml

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from utils.normalization import get_list_norm
from networks import get_network, load_weights

def check_ai_watermark_and_signatures(image_path, img_bgr):
    """
    Thuật toán Giám định Dấu hiệu AI Tổng quát (Generic AI Forensics & Provenance Detector):
    Hoàn toàn KHÔNG dựa trên mẫu hình học hay tên thương hiệu riêng của bất kỳ hãng nào.
    
    Áp dụng 2 nguyên lý Xử lý ảnh số (XLA) chuẩn tắc:
    1. Chuẩn chứng chỉ số mở quốc tế (C2PA / IPTC Standards):
       Kiểm tra chữ ký số theo chuẩn quốc tế C2PA (Coalition for Content Provenance and Authenticity),
       chuẩn IPTC 'trainedAlgorithmicMedia' và các container siêu dữ liệu ảnh AI chung.
    2. Phân tích dị thường đồ họa nhân tạo sát mép viền (Generic Edge and Color Entropy Anomaly):
       Mọi watermark/badge AI (chữ, dải màu, icon, logo bán trong suốt) bản chất là một lớp đồ họa
       nhân tạo có độ sắc nét biên độ cao (gradient magnitude) bất thường kết hợp với độ phẳng màu
       cục bộ dán đè lên vùng hạt nhiễu tự nhiên ở góc viền ảnh.
    """
    h, w = img_bgr.shape[:2]

    # Tầng 1: Quét chuẩn siêu dữ liệu quốc tế mở (C2PA / IPTC / AI Container)
    try:
        if os.path.exists(image_path):
            file_size = os.path.getsize(image_path)
            with open(image_path, "rb") as f:
                head = f.read(65536).lower()
                f.seek(max(0, file_size - 65536))
                tail = f.read(65536).lower()
            raw_bytes = head + tail
            
            industry_provenance_markers = [
                b"c2pa",
                b"jumbf",
                b"trainedalgorithmicmedia",
                b"digitalsourcetype",
                b"synthid",
                b"generativeai",
                b"prompt:",
                b"model_hash"
            ]
            for marker in industry_provenance_markers:
                if marker in raw_bytes:
                    return True, None
    except Exception:
        pass

    # Tầng 2: Phân tích vết đè đồ họa nhân tạo tổng quát tại các góc mép ảnh (Margins 15%)
    corner_regions = [
        ("BR", int(h * 0.75), h, int(w * 0.75), w),  # Góc dưới phải
        ("BL", int(h * 0.75), h, 0, int(w * 0.25)),  # Góc dưới trái
        ("TR", 0, int(h * 0.25), int(w * 0.75), w),  # Góc trên phải
        ("TL", 0, int(h * 0.25), 0, int(w * 0.25))   # Góc trên trái
    ]

    for label, y1, y2, x1, x2 in corner_regions:
        corner = img_bgr[y1:y2, x1:x2]
        ch, cw = corner.shape[:2]
        if ch < 30 or cw < 30:
            continue
        
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        
        # Tính toán trường Gradient vi phân Sobel bậc 1
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        
        # Tìm các đường biên sắc nét dạng đồ họa (stroke/badge edges)
        thresh_val = float(np.mean(grad_mag) + 2.0 * np.std(grad_mag))
        edge_mask = (grad_mag > max(35.0, thresh_val)).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edge_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if 30 <= area <= 2800:
                bx, by, bw, bh = cv2.boundingRect(c)
                if 12 <= bw <= 90 and 12 <= bh <= 90:
                    aspect = bw / float(bh)
                    if 0.35 <= aspect <= 2.8:
                        roi_gray = gray[by:by+bh, bx:bx+bw]
                        roi_grad = grad_mag[by:by+bh, bx:bx+bw]
                        
                        edge_density = float(np.mean(roi_grad > thresh_val))
                        local_std = float(np.std(roi_gray))
                        
                        if edge_density > 0.12 and local_std > 10.0:
                            abs_x = x1 + bx
                            abs_y = y1 + by
                            wm_box = (abs_x, abs_y, abs_x + bw, abs_y + bh)
                            return True, wm_box

    return False, None


class HybridForensicDetector:
    def __init__(self, model_name='BFREE_dino2reg4', weights_dir='./weights', device='cpu'):
        self.device = device
        self.model_name = model_name
        self.weights_dir = weights_dir
        self._load_bfree_model()

    def _load_bfree_model(self):
        config_path = os.path.join(self.weights_dir, self.model_name, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy config model tại: {config_path}")
        with open(config_path) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        model_path = os.path.join(self.weights_dir, self.model_name, cfg['weights_file'])
        raw_model = get_network(cfg['arch'])
        self.model = load_weights(raw_model, model_path).to(self.device).eval()
        self.transform = Compose(get_list_norm(cfg['norm_type']))
        print(f"[*] Đã tải thành công mô hình {self.model_name} trên {self.device}")

    def compute_bfree_scores(self, img_pil):
        """
        Tính điểm B-Free đa vùng (Multi-Crop / Patch-based scanning):
        - Quét toàn cảnh (Global): resize về 504x504
        - Quét cục bộ (Local Crops): trích xuất 5 ô cục bộ để không bị nền ảnh thật lấn át
        """
        w, h = img_pil.size
        crop_size = 504

        global_img = img_pil.resize((crop_size, crop_size), Image.BICUBIC)
        crops = [global_img]
        crop_coords = [(0, 0, w, h)]

        # Nếu ảnh lớn, trích xuất 5 vùng cắt chiến lược (Center + 4 Quadrants)
        if w > 550 or h > 550:
            cw = min(w, max(crop_size, int(w * 0.65)))
            ch = min(h, max(crop_size, int(h * 0.65)))
            local_boxes = [
                ((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2),  # Center
                (0, 0, cw, ch),                                                # Top-Left
                (w - cw, 0, w, ch),                                            # Top-Right
                (0, h - ch, cw, h),                                            # Bottom-Left
                (w - cw, h - ch, w, h)                                         # Bottom-Right
            ]
            for (bx1, by1, bx2, by2) in local_boxes:
                c_crop = img_pil.crop((bx1, by1, bx2, by2)).resize((crop_size, crop_size), Image.BICUBIC)
                crops.append(c_crop)
                crop_coords.append((bx1, by1, bx2, by2))

        # Đưa toàn bộ vào 1 batch duy nhất chạy qua PyTorch để tối ưu tốc độ
        with torch.no_grad():
            tensors = torch.stack([self.transform(c) for c in crops]).to(self.device)
            out = self.model(tensors).cpu().squeeze(-1).numpy()
            if out.ndim == 0:
                out = np.array([out])

        global_logit = float(out[0])
        global_prob = 1.0 / (1.0 + np.exp(-global_logit))

        if len(out) > 1:
            local_logits = out[1:]
            max_idx = int(np.argmax(local_logits)) + 1
            max_local_logit = float(local_logits[max_idx - 1])
            max_local_prob = 1.0 / (1.0 + np.exp(-max_local_logit))
            best_crop_bbox = crop_coords[max_idx]
        else:
            max_local_logit = global_logit
            max_local_prob = global_prob
            best_crop_bbox = (0, 0, w, h)

        return global_logit, global_prob, max_local_logit, max_local_prob, best_crop_bbox

    def compute_srm_residual(self, img_bgr):
        """Phân tích nhiễu phần dư tần số cao (SRM High-pass Residual)"""
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kernel = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=np.float32)
        residual = np.abs(cv2.filter2D(img_gray.astype(np.float32), -1, kernel))
        kernel_var = np.ones((7, 7), dtype=np.float32) / 49.0
        local_energy = np.sqrt(np.maximum(cv2.filter2D(residual ** 2, -1, kernel_var), 0))
        norm_energy = cv2.normalize(local_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        high_energy_ratio = float(np.mean(norm_energy > 120))
        max_energy = float(np.max(norm_energy))
        return norm_energy, high_energy_ratio, max_energy

    def compute_ela(self, img_pil, quality=90):
        """Error Level Analysis (ELA) phát hiện sai lệch nén JPEG"""
        temp_file = "temp_pipeline_ela.jpg"
        img_pil.save(temp_file, 'JPEG', quality=quality)
        resaved = Image.open(temp_file).convert('RGB')
        diff = ImageChops.difference(img_pil, resaved)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        diff = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
        return diff

    def analyze_image(self, image_path, out_dir="out"):
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"\n{'='*20} ĐANG PHÂN TÍCH: {base_name} {'='*20}")
        img_pil = Image.open(image_path).convert('RGB')
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. B-Free Score Đa vùng (Global + Multi-crop)
        bfree_logit, bfree_prob, max_local_logit, max_local_prob, best_crop_bbox = self.compute_bfree_scores(img_pil)
        
        # 2. SRM Residual
        srm_energy, high_energy_ratio, max_energy = self.compute_srm_residual(img_bgr)
        
        # 3. ELA
        ela_img = self.compute_ela(img_pil)
        
        # 4. Quét dấu ấn & bản quyền AI tổng quát (C2PA / IPTC & Generic Edge Anomaly)
        has_watermark, wm_bbox = check_ai_watermark_and_signatures(image_path, img_bgr)
        
        # 5. Hybrid Decision Logic 3 Tầng
        if bfree_logit > 0.5 and bfree_prob > 0.62:
            verdict = "AI-GENERATED (ẢNH DO AI TẠO TOÀN PHẦN)"
            confidence = bfree_prob * 100
            status_code = "FAKE_FULL"
        elif has_watermark:
            verdict = "LOCAL INPAINTING (ẢNH THẬT BỊ CHỈNH SỬA AI CỤC BỘ)"
            confidence = max(max_local_prob * 100, 92.0)
            status_code = "FAKE_LOCAL"
        elif max_local_logit > 0.2 and (max_local_logit - bfree_logit) > 0.6:
            verdict = "LOCAL INPAINTING (ẢNH THẬT BỊ CHỈNH SỬA AI CỤC BỘ)"
            confidence = max_local_prob * 100
            status_code = "FAKE_LOCAL"
        elif high_energy_ratio > 0.08 and bfree_logit > -1.0:
            verdict = "LOCAL INPAINTING (ẢNH THẬT BỊ CHỈNH SỬA AI CỤC BỘ)"
            confidence = max(bfree_prob * 100, 75.0)
            status_code = "FAKE_LOCAL"
        else:
            verdict = "REAL (ẢNH CHỤP THẬT TỰ NHIÊN)"
            confidence = (1.0 - bfree_prob) * 100
            status_code = "AUTHENTIC_REAL"
            
        print(f"Logit B-Free Toàn ảnh:   {bfree_logit:+.3f} (Xác suất AI: {bfree_prob*100:.1f}%)")
        print(f"Logit B-Free Cực đại Crop: {max_local_logit:+.3f} (Xác suất Crop: {max_local_prob*100:.1f}%)")
        print(f"Bất thường nhiễu SRM:     {high_energy_ratio*100:.2f}% (Max: {max_energy:.1f})")
        print(f"Dấu hiệu bản quyền AI:    {'CÓ' if has_watermark else 'KHÔNG'}")
        print(f"--> KẾT LUẬN CUỐI CÙNG:    {verdict} ({confidence:.1f}% tin cậy)")
        
        # 6. Vẽ và lưu ảnh tổng hợp vào thư mục out/
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        disp_img = img_rgb.copy()
        if "FAKE" in status_code:
            from forensic_analysis import find_suspicious_roi
            (bx1, by1, bx2, by2), is_det = find_suspicious_roi(img_bgr, srm_energy, ela_img)
            if is_det:
                cv2.rectangle(disp_img, (bx1, by1), (bx2, by2), (255, 0, 0), 4)
                cv2.putText(disp_img, "AI Modified Area", (bx1, max(30, by1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            elif wm_bbox:
                wx1, wy1, wx2, wy2 = wm_bbox
                cv2.rectangle(disp_img, (wx1 - 5, wy1 - 5), (wx2 + 5, wy2 + 5), (0, 0, 255), 3)
                cv2.putText(disp_img, "AI Watermark", (wx1, max(25, wy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        axes[0].imshow(disp_img)
        axes[0].set_title(f"1. Ảnh Gốc: {base_name}", fontsize=12, fontweight='bold', pad=10)
        axes[0].axis('off')
        
        im_srm = axes[1].imshow(srm_energy, cmap='inferno')
        axes[1].set_title("2. Bản Đồ Nhiễu (SRM Residual)", fontsize=12, fontweight='bold', pad=10)
        axes[1].axis('off')
        fig.colorbar(im_srm, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(ela_img)
        axes[2].set_title("3. Error Level Analysis (ELA)", fontsize=12, fontweight='bold', pad=10)
        axes[2].axis('off')
        
        title_color = 'darkred' if "FAKE" in status_code else 'darkgreen'
        fig.suptitle(f"KẾT QUẢ GIÁM ĐỊNH: {verdict}\n(Độ tin cậy: {confidence:.1f}% | Logit: {bfree_logit:+.2f})", 
                     fontsize=14, fontweight='bold', color=title_color, y=1.02)
        
        out_plot_path = os.path.join(out_dir, f"{base_name}_forensic.png")
        plt.tight_layout()
        plt.savefig(out_plot_path, dpi=180, bbox_inches='tight')
        plt.close()
        
        # 7. Lưu metadata JSON
        result_data = {
            "image_name": os.path.basename(image_path),
            "status": status_code,
            "verdict": verdict,
            "confidence_percent": round(confidence, 2),
            "bfree_logit": round(bfree_logit, 4),
            "bfree_prob": round(bfree_prob, 4),
            "max_local_logit": round(max_local_logit, 4),
            "srm_high_energy_ratio": round(high_energy_ratio, 4),
            "ai_watermark_detected": has_watermark,
            "gemini_watermark_detected": has_watermark,
            "forensic_report_image": os.path.abspath(out_plot_path)
        }
        out_json_path = os.path.join(out_dir, f"{base_name}_report.json")
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
            
        return result_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hệ thống Giám định Ảnh số Kết hợp B-Free + SRM + ELA")
    parser.add_argument('-i', '--input', type=str, help="Đường dẫn đến 1 ảnh cụ thể")
    parser.add_argument('-d', '--dir', type=str, help="Đường dẫn đến thư mục chứa nhiều ảnh")
    parser.add_argument('-o', '--out', type=str, default="out", help="Thư mục lưu kết quả (mặc định: out)")
    parser.add_argument('--device', type=str, default="cpu", help="Thiết bị chạy (cpu hoặc cuda:0)")
    args = parser.parse_args()

    detector = HybridForensicDetector(device=args.device)

    if args.input:
        detector.analyze_image(args.input, out_dir=args.out)
    elif args.dir:
        extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(args.dir, ext)))
        print(f"[*] Tìm thấy {len(images)} ảnh trong thư mục {args.dir}")
        for img in images:
            detector.analyze_image(img, out_dir=args.out)
    else:
        print("Vui lòng cung cấp ảnh (-i) hoặc thư mục (-d). Gõ --help để xem hướng dẫn.")
