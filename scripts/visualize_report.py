"""
visualize_report.py
───────────────────
为报告生成三类检测可视化图：
  Figure 1  基线（干净图 + Hard-NMS）       → vis_baseline/
  Figure 2  失效（干净 vs L=4噪声, Hard-NMS）→ vis_failure/
  Figure 3  改进（L=4 Hard-NMS vs Soft-NMS）→ vis_improvement/

Set SARDET_DATA_ROOT and MSFA_CHECKPOINT, then run this script from the
supplementary repository. Optional path overrides are documented in README.md.

依赖：mmdet, kymatio, matplotlib, opencv-python, numpy
"""

import os, json, random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')                         # 无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

import torch
from mmdet.apis import init_detector, inference_detector

# ═══════════════════════════════════════════════════════════
#  1.  Paths and parameters
# ═══════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parents[1]
SARDET_ROOT = Path(os.environ.get('SARDET_DATA_ROOT', '/path/to/SARDet_100K'))
CONFIG_HARDNMS = os.environ.get(
    'MSFA_HARD_CONFIG', str(REPO_ROOT / 'configs' / 'run_eval.py'))
CONFIG_SOFTNMS = os.environ.get(
    'MSFA_SOFT_CONFIG', str(REPO_ROOT / 'configs' / 'run_eval_softnms.py'))
CHECKPOINT = os.environ.get(
    'MSFA_CHECKPOINT', '/path/to/best_coco_bbox_mAP_epoch_12.pth')

DATA_ROOT  = str(SARDET_ROOT / 'JPEGImages')
TEST_CLEAN = os.path.join(DATA_ROOT, 'test')
TEST_L4    = os.path.join(DATA_ROOT, 'test_speckle_L4')
ANNO_FILE  = str(SARDET_ROOT / 'Annotations' / 'test.json')

OUTPUT_DIR = os.environ.get(
    'VIS_OUTPUT_DIR', str(REPO_ROOT / 'outputs' / 'visualizations'))
SCORE_THR       = 0.30   # 干净图可视化阈值
SCORE_THR_NOISY = 0.05   # 噪声图阈值（退化后模型置信度整体下移，需放宽）
N_IMAGES   = 6             # 每类图展示的图片数量
DEVICE     = 'cuda:0'

# ═══════════════════════════════════════════════════════════
#  2.  类别与颜色
# ═══════════════════════════════════════════════════════════
CLASSES = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']

# BGR 格式（OpenCV），同时用于 matplotlib（归一化到0-1）
COLORS_BGR = {
    'ship':     (219, 152,  52),   # blue
    'aircraft': (  0,165, 255),   # orange
    'car':      ( 50,205,  50),   # green
    'tank':     ( 60,  20, 220),  # red
    'bridge':   (180,  30, 145),  # purple
    'harbor':   (205,184,  68),   # teal
}

def bgr_to_rgb01(bgr):
    return (bgr[2]/255, bgr[1]/255, bgr[0]/255)


# ═══════════════════════════════════════════════════════════
#  3.  工具函数
# ═══════════════════════════════════════════════════════════

def select_images(anno_file, n=6, seed=42):
    """
    从 test.json 中挑选 n 张具有代表性的图片：
      - 至少含 3 个标注框
      - 不超过 20 个标注框（避免图面太密）
      - 尽量覆盖多种类别
    返回图片文件名列表（不含路径）。
    """
    with open(anno_file) as f:
        coco = json.load(f)

    # 统计每张图的标注数量与类别集合
    img_info = {img['id']: {'file_name': img['file_name'], 'cats': set()}
                for img in coco['images']}
    img_count = {img['id']: 0 for img in coco['images']}

    for ann in coco['annotations']:
        iid = ann['image_id']
        if iid in img_count:
            img_count[iid] += 1
            img_info[iid]['cats'].add(ann['category_id'])

    # 筛选
    candidates = [
        info for iid, info in img_info.items()
        if 3 <= img_count[iid] <= 20
    ]

    # 按类别数量降序排，再随机抽
    candidates.sort(key=lambda x: -len(x['cats']))
    rng = random.Random(seed)
    selected = candidates[:max(n * 4, 40)]   # 先取前段高质量候选
    rng.shuffle(selected)
    chosen = selected[:n]
    return [c['file_name'] for c in chosen]


def find_images_with_detections(model, directory, n=4,
                                score_thr=0.001, max_scan=500):
    """
    扫描目录，找到模型在该图上有检测结果的图片。
    用于失效/改进对比图的选图——确保噪声图上能看到（低置信度）检测框。
    """
    imgs = (sorted(Path(directory).glob('*.jpg')) +
            sorted(Path(directory).glob('*.png')) +
            sorted(Path(directory).glob('*.bmp')))

    found = []
    print(f'  扫描 {directory}（最多 {max_scan} 张）...')
    for p in imgs[:max_scan]:
        r  = inference_detector(model, str(p))
        sc = r.pred_instances.scores.cpu().numpy()
        n_det = int((sc >= score_thr).sum())
        if n_det >= 1:
            print(f'  ✓ {p.name}: det={n_det}, max_score={sc.max():.3f}')
            found.append(p.name)
            if len(found) >= n:
                break
    if not found:
        print(f'  [WARN] 扫描 {max_scan} 张后仍无检测，降低 score_thr 或检查噪声数据')
    return found


def find_image(fname, directory):
    """在目录中查找文件（支持 .jpg/.png/.bmp 扩展名替换）"""
    # 直接找
    p = os.path.join(directory, fname)
    if os.path.exists(p):
        return p
    # 尝试替换扩展名
    stem = Path(fname).stem
    for ext in ['.jpg', '.png', '.bmp']:
        p2 = os.path.join(directory, stem + ext)
        if os.path.exists(p2):
            return p2
    return None


def run_inference(model, img_path):
    """
    运行推理，返回 (bboxes, scores, labels) numpy 数组。
    bboxes: (N,4) xyxy 格式
    """
    result = inference_detector(model, img_path)
    pi = result.pred_instances
    bboxes = pi.bboxes.cpu().numpy()
    scores = pi.scores.cpu().numpy()
    labels = pi.labels.cpu().numpy()
    return bboxes, scores, labels


def draw_boxes_on_ax(ax, img_rgb, bboxes, scores, labels,
                     score_thr=SCORE_THR, title=''):
    """在 matplotlib Axes 上绘制检测框"""
    ax.imshow(img_rgb, cmap='gray' if img_rgb.ndim == 2 else None)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis('off')

    mask = scores >= score_thr
    for box, sc, lb in zip(bboxes[mask], scores[mask], labels[mask]):
        cls_name = CLASSES[lb]
        color = bgr_to_rgb01(COLORS_BGR[cls_name])
        x1, y1, x2, y2 = box
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle='square,pad=0',
            linewidth=1.5,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, f'{cls_name} {sc:.2f}',
                color='white', fontsize=5.5,
                bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor='none'))

    n_det = int(mask.sum())
    ax.text(5, img_rgb.shape[0] - 5,
            f'det={n_det}', color='yellow',
            fontsize=7, va='bottom',
            bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))


def load_img_rgb(path):
    """读取图片并转为 RGB"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'无法读取图片：{path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_legend():
    """生成类别图例的 patch 列表"""
    return [
        mpatches.Patch(color=bgr_to_rgb01(COLORS_BGR[c]), label=c)
        for c in CLASSES
    ]


# ═══════════════════════════════════════════════════════════
#  4.  三类图生成函数
# ═══════════════════════════════════════════════════════════

def fig_baseline(fnames, model_hard, out_dir):
    """
    Figure 1：基线图（干净图 + Hard-NMS）
    布局：2行 × 3列，共6张图
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(fnames)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.2))
    fig.suptitle('Figure 1  Baseline Detection (Clean Images, Hard-NMS)',
                 fontsize=12, fontweight='bold', y=1.01)
    axes = np.array(axes).flatten()

    for i, fname in enumerate(fnames):
        path = find_image(fname, TEST_CLEAN)
        if path is None:
            print(f'[WARN] 找不到 {fname}，跳过')
            axes[i].axis('off')
            continue
        img = load_img_rgb(path)
        bboxes, scores, labels = run_inference(model_hard, path)
        stem = Path(fname).stem
        draw_boxes_on_ax(axes[i], img, bboxes, scores, labels,
                         title=f'{stem}')

    # 隐藏多余子图
    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.legend(handles=make_legend(), loc='lower center',
               ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'fig1_baseline.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 基线图已保存：{out_path}')


def fig_failure(fnames, model_hard, out_dir):
    """
    Figure 2：失效图（干净 vs L=4噪声，均用 Hard-NMS）
    布局：N行 × 2列（左=干净，右=L4）
    """
    os.makedirs(out_dir, exist_ok=True)
    # 只取前4张，避免图太高
    fnames = fnames[:4]
    n = len(fnames)

    fig, axes = plt.subplots(n, 2,
                             figsize=(9, n * 4.0))
    fig.suptitle(
        'Figure 2  Failure Case: Speckle Noise Degradation (L=4)\n'
        'Left: Clean  |  Right: Gamma(4, 0.25) multiplicative noise',
        fontsize=11, fontweight='bold', y=1.01
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, fname in enumerate(fnames):
        path_clean = find_image(fname, TEST_CLEAN)
        path_l4    = find_image(fname, TEST_L4)

        if path_clean is None or path_l4 is None:
            print(f'[WARN] 找不到 {fname}，跳过')
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            continue

        img_clean = load_img_rgb(path_clean)
        img_l4    = load_img_rgb(path_l4)

        bb_c, sc_c, lb_c = run_inference(model_hard, path_clean)
        bb_l, sc_l, lb_l = run_inference(model_hard, path_l4)

        n_clean = int((sc_c >= SCORE_THR).sum())
        n_l4    = int((sc_l >= SCORE_THR).sum())
        delta   = n_l4 - n_clean

        draw_boxes_on_ax(axes[i, 0], img_clean, bb_c, sc_c, lb_c,
                         score_thr=SCORE_THR,
                         title=f'Clean  (det={n_clean})')
        draw_boxes_on_ax(axes[i, 1], img_l4,    bb_l, sc_l, lb_l,
                         score_thr=SCORE_THR_NOISY,
                         title=f'L=4 Noise  (det={n_l4}, Δ={delta:+d})')

    fig.legend(handles=make_legend(), loc='lower center',
               ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'fig2_failure.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 失效图已保存：{out_path}')


def fig_improvement(fnames, model_hard, model_soft, out_dir):
    """
    Figure 3：改进图（L=4噪声，Hard-NMS vs Soft-NMS）
    布局：N行 × 2列（左=Hard-NMS，右=Soft-NMS）
    """
    os.makedirs(out_dir, exist_ok=True)
    fnames = fnames[:4]
    n = len(fnames)

    fig, axes = plt.subplots(n, 2,
                             figsize=(9, n * 4.0))
    fig.suptitle(
        'Figure 3  Improvement: Hard-NMS vs Soft-NMS (L=4 Noisy Images)\n'
        'Left: Hard-NMS  |  Right: Soft-NMS',
        fontsize=11, fontweight='bold', y=1.01
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, fname in enumerate(fnames):
        path_l4 = find_image(fname, TEST_L4)
        if path_l4 is None:
            print(f'[WARN] 找不到 L4 图片 {fname}，跳过')
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            continue

        img_l4 = load_img_rgb(path_l4)
        bb_h, sc_h, lb_h = run_inference(model_hard, path_l4)
        bb_s, sc_s, lb_s = run_inference(model_soft, path_l4)

        n_hard = int((sc_h >= SCORE_THR).sum())
        n_soft = int((sc_s >= SCORE_THR).sum())
        delta  = n_soft - n_hard

        draw_boxes_on_ax(axes[i, 0], img_l4, bb_h, sc_h, lb_h,
                         score_thr=SCORE_THR_NOISY,
                         title=f'Hard-NMS  (det={n_hard})')
        draw_boxes_on_ax(axes[i, 1], img_l4, bb_s, sc_s, lb_s,
                         score_thr=SCORE_THR_NOISY,
                         title=f'Soft-NMS  (det={n_soft}, Δ={delta:+d})')

    fig.legend(handles=make_legend(), loc='lower center',
               ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'fig3_improvement.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 改进图已保存：{out_path}')


def fig_three_panel(fnames, model_hard, model_soft, out_dir):
    """
    Figure 4（可选）：三栏对比图（干净 | L4+Hard | L4+Soft）
    选前3张图，最直观展示完整流程。
    """
    os.makedirs(out_dir, exist_ok=True)
    fnames = fnames[:3]
    n = len(fnames)

    fig, axes = plt.subplots(n, 3, figsize=(13, n * 4.0))
    fig.suptitle(
        'Figure 4  Three-Panel Comparison\n'
        'Clean + Hard-NMS  |  L=4 Noise + Hard-NMS  |  L=4 Noise + Soft-NMS',
        fontsize=11, fontweight='bold', y=1.01
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, fname in enumerate(fnames):
        path_c  = find_image(fname, TEST_CLEAN)
        path_l4 = find_image(fname, TEST_L4)

        if path_c is None or path_l4 is None:
            for j in range(3):
                axes[i, j].axis('off')
            continue

        img_c  = load_img_rgb(path_c)
        img_l4 = load_img_rgb(path_l4)

        bb_c,  sc_c,  lb_c  = run_inference(model_hard, path_c)
        bb_lh, sc_lh, lb_lh = run_inference(model_hard, path_l4)
        bb_ls, sc_ls, lb_ls = run_inference(model_soft, path_l4)

        draw_boxes_on_ax(axes[i, 0], img_c,  bb_c,  sc_c,  lb_c,
                         score_thr=SCORE_THR,
                         title=f'Clean + Hard-NMS')
        draw_boxes_on_ax(axes[i, 1], img_l4, bb_lh, sc_lh, lb_lh,
                         score_thr=SCORE_THR_NOISY,
                         title=f'L=4 + Hard-NMS')
        draw_boxes_on_ax(axes[i, 2], img_l4, bb_ls, sc_ls, lb_ls,
                         score_thr=SCORE_THR_NOISY,
                         title=f'L=4 + Soft-NMS ✓')

    fig.legend(handles=make_legend(), loc='lower center',
               ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'fig4_three_panel.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 三栏对比图已保存：{out_path}')


def fig_metric_comparison(out_dir):
    """
    Figure 3：Hard-NMS vs Soft-NMS 定量对比条形图
    ─────────────────────────────────────────────
    用实验数值直接绘制，无需模型推理。
    ⚠️  Soft-NMS L=4 噪声的结果跑完后请更新 RESULTS 字典。
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── 实验结果（从评估日志填入）────────────────────────────────
    # Soft-NMS Clean 使用修正噪声前的有效结果（clean test set 未变动）
    # Soft-NMS L=4 为修正物理正确噪声后的新实验结果
    RESULTS = {
        'Hard-NMS\n(Clean)':  dict(mAP=0.511, AP50=0.839, AP75=0.547),
        'Soft-NMS\n(Clean)':  dict(mAP=0.529, AP50=0.848, AP75=0.578),
        'Hard-NMS\n(L=4)':    dict(mAP=0.150, AP50=0.265, AP75=0.154),
        'Soft-NMS\n(L=4)':    dict(mAP=0.162, AP50=0.282, AP75=0.169),
    }
    PLACEHOLDER = None   # 所有条目均已填入

    metrics  = ['mAP', 'AP50', 'AP75']
    labels   = ['mAP (IoU 0.5:0.95)', 'AP@50', 'AP@75']
    colors   = {
        'Hard-NMS\n(Clean)': '#4472C4',
        'Soft-NMS\n(Clean)': '#70AD47',
        'Hard-NMS\n(L=4)':   '#FF4444',
        'Soft-NMS\n(L=4)':   '#FF9800',
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        'Quantitative Comparison: Hard-NMS vs Soft-NMS Configurations\n'
        'Baseline (clean images) and Failure Scenario (L=4 speckle noise)',
        fontsize=11, fontweight='bold'
    )

    conditions = list(RESULTS.keys())
    x = np.arange(len(conditions))
    width = 0.55

    for ax, metric, label in zip(axes, metrics, labels):
        vals   = [RESULTS[c][metric] for c in conditions]
        bars   = ax.bar(x, vals, width,
                        color=[colors[c] for c in conditions],
                        edgecolor='white', linewidth=0.8)

        # 数值标注
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold')

        # 标出噪声造成的下降，以及 L=4 条件下 Soft 配置的相对提升。
        idx_h = conditions.index('Hard-NMS\n(L=4)')
        idx_s = conditions.index('Soft-NMS\n(L=4)')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=9, pad=4)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

        # 噪声场景下降幅标注
        v_clean = RESULTS['Hard-NMS\n(Clean)'][metric]
        v_noisy = RESULTS['Hard-NMS\n(L=4)'][metric]
        drop_pct = (v_noisy - v_clean) / v_clean * 100
        ax.annotate(
            f'Δ={drop_pct:+.1f}%\n(noise)',
            xy=(idx_h, v_noisy), xytext=(idx_h + 0.35, v_noisy + 0.12),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
            fontsize=7.5, color='red'
        )

        soft_gain = (
            RESULTS['Soft-NMS\n(L=4)'][metric] - v_noisy
        ) / v_noisy * 100
        ax.annotate(
            f'+{soft_gain:.1f}%',
            xy=(idx_s, RESULTS['Soft-NMS\n(L=4)'][metric]),
            xytext=(idx_s - 0.35, RESULTS['Soft-NMS\n(L=4)'][metric] + 0.05),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.0),
            fontsize=7.5, color='#2E7D32'
        )

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'fig3_metric_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 定量对比图已保存：{out_path}')


# ═══════════════════════════════════════════════════════════
#  5.  主入口
# ═══════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 步骤 1：选取代表性图片 ────────────────────────────
    print('正在从标注文件中选取代表性图片...')
    fnames = select_images(ANNO_FILE, n=N_IMAGES)
    print(f'已选取 {len(fnames)} 张图片：{fnames}')

    # ── 步骤 2：加载模型 ──────────────────────────────────
    print('\n加载 Hard-NMS 模型...')
    model_hard = init_detector(CONFIG_HARDNMS, CHECKPOINT, device=DEVICE)
    model_hard.eval()
    # roi_head.test_cfg 存储 rcnn 部分（score_thr/nms/max_per_img）
    # 降低 score_thr 让推理返回低置信度框，可视化用
    model_hard.roi_head.test_cfg.score_thr = 0.001
    model_hard.roi_head.test_cfg.max_per_img = 300

    print('加载 Soft-NMS 模型...')
    model_soft = init_detector(CONFIG_SOFTNMS, CHECKPOINT, device=DEVICE)
    model_soft.eval()
    model_soft.roi_head.test_cfg.score_thr = 0.001
    model_soft.roi_head.test_cfg.max_per_img = 300

    # ── 步骤 3：生成各类图 ────────────────────────────────
    print('\n生成 Figure 1：基线检测图...')
    fig_baseline(fnames, model_hard,
                 out_dir=os.path.join(OUTPUT_DIR, 'vis_baseline'))

    # Figure 2 用标注多的图（fnames）：干净图det多，噪声图det=0，对比强烈
    print('\n生成 Figure 2：失效案例图...')
    fig_failure(fnames, model_hard,
                out_dir=os.path.join(OUTPUT_DIR, 'vis_failure'))

    # Figure 3 改为定量对比图（Hard-NMS vs Soft-NMS 各指标柱状图）
    print('\n生成 Figure 3：Hard-NMS vs Soft-NMS 定量对比图...')
    fig_metric_comparison(out_dir=os.path.join(OUTPUT_DIR, 'vis_improvement'))

    # Figure 4 扫描噪声下有检测的图做三栏对比（展示噪声下两种NMS的细微差别）
    print('\n扫描 L=4 噪声图，寻找有检测结果的图片...')
    fnames_noisy = find_images_with_detections(
        model_hard, TEST_L4, n=3, score_thr=SCORE_THR_NOISY)
    if not fnames_noisy:
        print('[WARN] 未找到有检测的噪声图，跳过 Figure 4')
    else:
        print('\n生成 Figure 4：三栏综合对比图...')
        fig_three_panel(fnames_noisy, model_hard, model_soft,
                        out_dir=os.path.join(OUTPUT_DIR, 'vis_three_panel'))

    # ── 步骤 4：汇总 ──────────────────────────────────────
    print(f'\n全部完成！输出目录：{OUTPUT_DIR}')
    print('  vis_baseline/fig1_baseline.png    ← 报告第4节（复现）')
    print('  vis_failure/fig2_failure.png      ← 报告第5节（失效分析）')
    print('  vis_improvement/fig3_improvement.png ← 报告第6-7节（改进）')
    print('  vis_three_panel/fig4_three_panel.png ← 报告最佳总图')


if __name__ == '__main__':
    main()
