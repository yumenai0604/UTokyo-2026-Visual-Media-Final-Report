"""
准备两类失效测试所需的数据
Case 1: 相干斑噪声降质图像集
Case 2: 按来源子集分割标注文件
"""
import os, json, cv2, numpy as np
from pathlib import Path
from collections import defaultdict
 
DATA_ROOT = '/root/autodl-tmp/sardet100k/SARDet_100K/'
VAL_DIR   = DATA_ROOT + 'JPEGImages/test/'
ANN_FILE  = DATA_ROOT + 'Annotations/test.json'
 
#── Case 1: 相干斑噪声 ──────────────────────────────────────
def add_speckle(img, L):
    """Gamma分布乘性噪声，物理准确的SAR相干斑模型
    SAR图像本质为单通道灰度，同一像素位置的噪声实现唯一，
    不应在通道间独立采样（否则产生虚假彩色伪影）。
    修复：使用 img.shape[:2] 生成 (H,W) 噪声图，再广播至三通道。
    """
    noise_2d = np.random.gamma(shape=L, scale=1.0/L, size=img.shape[:2])  # (H, W)
    noise    = noise_2d[:, :, np.newaxis]                                   # (H, W, 1) → 广播至所有通道
    return np.clip(img.astype(np.float64) * noise, 0, 255).astype(np.uint8)
 
np.random.seed(42)
# 修复：补充 .bmp 格式（SARDet-100K 中部分图像为 BMP）
img_files = (list(Path(VAL_DIR).glob('*.jpg')) +
             list(Path(VAL_DIR).glob('*.png')) +
             list(Path(VAL_DIR).glob('*.bmp')))
print(f'[Case 1] 找到 {len(img_files)} 张验证图像')
 
for L in [1, 2, 4]:
    out_dir = Path(DATA_ROOT + f'JPEGImages/test_speckle_L{L}/')
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        cv2.imwrite(str(out_dir / img_path.name), add_speckle(img, L))
    print(f'  L={L}: {len(img_files)} 张 → {out_dir}')
 
# ── Case 2: 跨来源子集 ──────────────────────────────────────
with open(ANN_FILE) as f:
    coco = json.load(f)
 
print(f'\n[Case 2] 示例文件名: {coco["images"][0]["file_name"]}')
 
# 按文件名前缀分组（SARDet-100K各子数据集有不同前缀）
source_groups = defaultdict(list)
for img in coco['images']:
    # 去掉数字编号，保留字母前缀
    prefix = ''.join(c for c in img['file_name'].split('_')[0] if not c.isdigit())
    source_groups[prefix].append(img['id'])
 
print(f'发现 {len(source_groups)} 个来源:')
for src, ids in sorted(source_groups.items(), key=lambda x: -len(x[1])):
    print(f'  {src:20s}: {len(ids):5d} 张')
 
out_dir = DATA_ROOT + 'Annotations/subsets/'
os.makedirs(out_dir, exist_ok=True)
 
generated = []
for source, img_ids in source_groups.items():
    if len(img_ids) < 20:   # 跳过过小的子集
        continue
    img_id_set = set(img_ids)
    sub_images = [img for img in coco['images'] if img['id'] in img_id_set]
    sub_anns   = [ann for ann in coco['annotations'] if ann['image_id'] in img_id_set]
    sub_coco   = {'info': coco.get('info', {}),
                  'categories': coco['categories'],
                  'images': sub_images,
                  'annotations': sub_anns}
    out_path = f'{out_dir}val_{source}.json'
    with open(out_path, 'w') as f:
        json.dump(sub_coco, f)
    generated.append(source)
    print(f'  生成: val_{source}.json ({len(sub_images)} 张, {len(sub_anns)} 标注)')
 
print(f'\n数据准备完成，共生成 {len(generated)} 个子集')
 