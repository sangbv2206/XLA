import os
import sys
import yaml
import argparse
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from utils.normalization import get_list_norm
from networks import get_network, load_weights

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def get_config(model_name='BFREE_dino2reg4', weights_dir='./weights'):
    config_file = os.path.join(weights_dir, model_name, 'config.yaml')
    with open(config_file) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type']

def create_hann_window(size=504):
    w1d = np.hanning(size)
    w2d = np.outer(w1d, w1d)
    w2d = np.maximum(w2d, 0.05) # avoid 0 near boundaries
    return w2d

def detect_and_localize(
    image_path, 
    output_path='localized_result.png', 
    model_name='BFREE_dino2reg4', 
    device='cpu',
    crop_size=504,
    stride_x=86,
    stride_y=124,
    batch_size=7
):
    print(f"[*] Loading model {model_name} on {device}...")
    _, model_path, arch, norm_type = get_config(model_name)
    raw_model = get_network(arch)
    model = load_weights(raw_model, model_path).to(device).eval()
    transform = Compose(get_list_norm(norm_type))

    # Read image
    print(f"[*] Reading image: {image_path}")
    orig_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = orig_img.size
    print(f"[*] Image size: {orig_w}x{orig_h} (WxH)")

    # Prepare coordinate grid for sliding window
    if orig_w < crop_size or orig_h < crop_size:
        # Scale up if smaller than crop size
        scale = max(crop_size / orig_w, crop_size / orig_h)
        new_w, new_h = int(np.ceil(orig_w * scale)), int(np.ceil(orig_h * scale))
        img = orig_img.resize((new_w, new_h), Image.BICUBIC)
        w, h = new_w, new_h
    else:
        img = orig_img
        w, h = orig_w, orig_h

    xs = list(range(0, w - crop_size + 1, stride_x))
    if xs[-1] != w - crop_size:
        xs.append(w - crop_size)
    ys = list(range(0, h - crop_size + 1, stride_y))
    if ys[-1] != h - crop_size:
        ys.append(h - crop_size)

    total_crops = len(xs) * len(ys)
    print(f"[*] Grid positions: {len(xs)} x {len(ys)} = {total_crops} crops (crop_size={crop_size})")

    # Arrays to accumulate heatmap
    accum_prob = np.zeros((h, w), dtype=np.float32)
    accum_logit = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    window = create_hann_window(crop_size)

    # Collect batches
    crops_list = []
    coords_list = []
    
    for y in ys:
        for x in xs:
            crop = img.crop((x, y, x + crop_size, y + crop_size))
            tensor_crop = transform(crop)
            crops_list.append(tensor_crop)
            coords_list.append((x, y))

    print(f"[*] Running model inference over {total_crops} crops...")
    with torch.no_grad():
        for i in range(0, total_crops, batch_size):
            batch_tensors = torch.stack(crops_list[i : i + batch_size]).to(device)
            # Patch embed projection
            p_emb = model.patch_embed.proj(batch_tensors)
            if model.patch_embed.flatten:
                p_emb = p_emb.flatten(2).transpose(1, 2)
            p_emb = model.patch_embed.norm(p_emb)
            
            # Classifier head
            logits = model.model(p_emb).cpu().squeeze(-1).numpy()
            if logits.ndim == 0:
                logits = np.array([logits])
            
            probs = 1.0 / (1.0 + np.exp(-logits))

            for j, (cx, cy) in enumerate(coords_list[i : i + batch_size]):
                lg = logits[j]
                pr = probs[j]
                accum_prob[cy : cy + crop_size, cx : cx + crop_size] += pr * window
                accum_logit[cy : cy + crop_size, cx : cx + crop_size] += lg * window
                weight_map[cy : cy + crop_size, cx : cx + crop_size] += window

    # Normalize by accumulated weights
    weight_map = np.maximum(weight_map, 1e-6)
    final_prob = accum_prob / weight_map
    final_logit = accum_logit / weight_map

    # Resize back to original dimensions if resized
    if (w, h) != (orig_w, orig_h):
        final_prob = cv2.resize(final_prob, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        final_logit = cv2.resize(final_logit, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Statistics
    max_logit = float(np.max(final_logit))
    max_prob = float(np.max(final_prob))
    avg_logit = float(np.mean(final_logit))
    avg_prob = float(np.mean(final_prob))

    print("================== KẾT QUẢ ĐỊNH VỊ ==================")
    print(f"Logit cao nhất toàn ảnh: {max_logit:+.3f}")
    print(f"Xác suất AI cao nhất:    {max_prob * 100:.1f}%")
    print(f"Logit trung bình:        {avg_logit:+.3f}")

    # Threshold for fake detection
    is_fake = max_logit > 0.0 or max_prob > 0.5
    verdict = "PHÁT HIỆN LÀM GIẢ CỤC BỘ (LOCAL INPAINTING / FAKE)" if is_fake else "ẢNH THẬT (REAL)"
    print(f"Kết luận: {verdict}")

    # Create visualization
    # 1. Heatmap RGB
    norm_prob = np.clip(final_prob, 0.0, 1.0)
    # Apply JET or TURBO colormap
    heatmap_u8 = (norm_prob * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 2. Overlay
    orig_np = np.array(orig_img)
    alpha = 0.45
    overlay = cv2.addWeighted(orig_np, 1 - alpha, heatmap_color, alpha, 0)

    # 3. Find bounding box of highest suspicion region
    binary_mask = (norm_prob > 0.50).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = None
    if len(contours) > 0:
        # Find largest contour
        largest_c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_c) > 500: # significant area
            bx, by, bw, bh = cv2.boundingRect(largest_c)
            bbox = (bx, by, bx + bw, by + bh)
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 0, 0), 3)
            cv2.putText(
                overlay, 
                f"AI Inpainting ({max_prob*100:.1f}%)", 
                (bx, max(30, by - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 0, 0), 
                2
            )

    # Plot 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f"B-Free AI Manipulation Localization: {verdict}\n(Max Confidence: {max_prob*100:.1f}%, Max Logit: {max_logit:+.2f})", fontsize=16, fontweight='bold', color='darkred' if is_fake else 'darkgreen')

    axes[0].imshow(orig_img)
    axes[0].set_title("1. Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    im1 = axes[1].imshow(norm_prob, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("2. AI Probability Heatmap", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Fake Probability (0.0 = Real, 1.0 = AI)", fontsize=11)

    axes[2].imshow(overlay)
    axes[2].set_title("3. Detected Fake Region (Overlay & Box)", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved localization visualization to: {output_path}")

    return {
        'verdict': verdict,
        'is_fake': is_fake,
        'max_logit': max_logit,
        'max_prob': max_prob,
        'bbox': bbox,
        'output_path': os.path.abspath(output_path)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help="Input image path")
    parser.add_argument('--output', '-o', type=str, default='localized_result.png', help="Output visualization path")
    parser.add_argument('--device', '-d', type=str, default='cpu', help="Torch device")
    args = parser.parse_args()

    detect_and_localize(args.input, args.output, device=args.device)
