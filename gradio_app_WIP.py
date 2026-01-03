import sys
sys.path.append('/DDColor') 

import argparse
import cv2
import numpy as np
import os
import torch
from basicsr.archs.ddcolor_arch import DDColor
import torch.nn.functional as F
import gradio as gr
import uuid

# --- CẤU HÌNH MODEL ---
model_path = 'experiments/train_ViCoW_1.2_0.5_freeze_ED_g_pretrain/models/net_g_20000.pth'  # Đường dẫn tới model đã huấn luyện
input_size = 512
model_size = 'large'

# --- CLASS XỬ LÝ CHÍNH (GIỮ NGUYÊN) ---
class ImageColorizationPipeline(object):
    def __init__(self, model_path, input_size=256, model_size='large'):
        self.input_size = input_size
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if model_size == 'tiny':
            self.encoder_name = 'convnext-t'
        else:
            self.encoder_name = 'convnext-l'

        self.model = DDColor(
            encoder_name=self.encoder_name,
            decoder_name='MultiScaleColorDecoder',
            input_size=[self.input_size, self.input_size],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['params'],
            strict=False)
        self.model.eval()

    @torch.no_grad()
    def process(self, img):
        self.height, self.width = img.shape[:2]
        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1] 

        img = cv2.resize(img, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor_gray_rgb).cpu() 

        output_ab_resize = F.interpolate(output_ab, size=(self.height, self.width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        
        output_img = (output_bgr * 255.0).round().astype(np.uint8)
        return output_img

# --- KHỞI TẠO MODEL ---
print("Đang tải model...")
colorizer = ImageColorizationPipeline(model_path=model_path,
                                      input_size=input_size,
                                      model_size=model_size)
print("Model đã sẵn sàng!")

# --- HÀM XỬ LÝ (ĐƠN GIẢN HÓA) ---
def colorize(img_rgb):
    if img_rgb is None:
        return None
    
    print("Đang xử lý...")
    
    # 1. Chuyển input từ RGB sang BGR (để đúng màu với model)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 2. Chạy model
    image_out_bgr = colorizer.process(img_bgr)
    
    # 3. Chuyển output từ BGR sang RGB (để hiển thị đúng trên web)
    image_out_rgb = cv2.cvtColor(image_out_bgr, cv2.COLOR_BGR2RGB)
    
    print("Xong!")
    # Trả về trực tiếp dữ liệu ảnh (Gradio Image gốc xử lý cái này cực tốt)
    return image_out_rgb

# --- GIAO DIỆN MỚI (CHUẨN 2 KHUNG HÌNH) ---
with gr.Blocks() as demo:
    gr.Markdown("### Tô màu ảnh đen trắng với DDColor")
    
    with gr.Row():
        # Cột bên trái: Input
        with gr.Column():
            input_img = gr.Image(label="Ảnh đen trắng (Input)", type="numpy")
            btn = gr.Button("Tô màu ngay", variant="primary")
            
        # Cột bên phải: Output (Dùng gr.Image chuẩn thay vì Slider)
        with gr.Column():
            output_img = gr.Image(label="Ảnh màu (Output)", type="numpy")

    # Kết nối nút bấm
    btn.click(fn=colorize, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch(share=True)