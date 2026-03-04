import os
import cv2
import numpy as np
import json
import math
import random
from skimage.morphology import skeletonize

# === 文件夹路径配置 ===
MASK_DIR = 'masks'  # UNet 掩码图所在文件夹
IMG_DIR = 'images'  # 原图所在文件夹
OUT_LINES_DIR = 'lines'  # 可视化图像保存文件夹
OUT_JSONS_DIR = 'jsons'  # JSON 保存文件夹


def get_line_params(x1, y1, x2, y2):
    """获取线段的角度和长度"""
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 180
    length = math.hypot(x2 - x1, y2 - y1)
    return angle, length


def dist_point_to_line(px, py, x1, y1, x2, y2):
    """计算点到直线的垂直距离"""
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.hypot(x2 - x1, y2 - y1)
    return num / (den + 1e-6)


def extract_lines_from_mask(mask: np.ndarray) -> list:
    """Extract merged line segments from a binary/grayscale mask."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    skeleton = skeletonize(binary > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=100,
    )

    merged_lines = []
    if lines is not None:
        lines_list = [l[0] for l in lines]
        lines_list.sort(key=lambda l: get_line_params(*l)[1], reverse=True)

        for line in lines_list:
            x1, y1, x2, y2 = line
            angle1, _ = get_line_params(x1, y1, x2, y2)

            merged = False
            for i, m_line in enumerate(merged_lines):
                mx1, my1, mx2, my2 = m_line
                angle2, _ = get_line_params(mx1, my1, mx2, my2)

                ang_diff = abs(angle1 - angle2)
                if ang_diff > 90:
                    ang_diff = 180 - ang_diff

                if ang_diff < 5:
                    d1 = dist_point_to_line(x1, y1, mx1, my1, mx2, my2)
                    d2 = dist_point_to_line(x2, y2, mx1, my1, mx2, my2)

                    if d1 < 10 and d2 < 10:
                        pts = [(x1, y1), (x2, y2), (mx1, my1), (mx2, my2)]
                        max_d = 0
                        best_pair = (pts[0], pts[1])
                        for pi in range(4):
                            for pj in range(pi + 1, 4):
                                d = math.hypot(pts[pi][0] - pts[pj][0], pts[pi][1] - pts[pj][1])
                                if d > max_d:
                                    max_d = d
                                    best_pair = (pts[pi], pts[pj])
                        merged_lines[i] = [best_pair[0][0], best_pair[0][1], best_pair[1][0], best_pair[1][1]]
                        merged = True
                        break
            if not merged:
                merged_lines.append([x1, y1, x2, y2])

    merged_lines = [line for line in merged_lines if get_line_params(*line)[1] > 30]
    return merged_lines


def build_line_features(merged_lines: list) -> list:
    features = []
    for idx, line in enumerate(merged_lines):
        x1, y1, x2, y2 = map(int, line)
        features.append(
            {
                "type": "LineString",
                "properties": {"id": idx + 1},
                "coordinates": [[float(x1), float(y1)], [float(x2), float(y2)]],
            }
        )
    return features


def main() -> None:
    # 自动创建输出文件夹（如果不存在）
    os.makedirs(OUT_LINES_DIR, exist_ok=True)
    os.makedirs(OUT_JSONS_DIR, exist_ok=True)

    mask_files = [
        f
        for f in os.listdir(MASK_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if not mask_files:
        print(f"在 '{MASK_DIR}' 文件夹中没有找到图片文件，请检查路径。")
        return

    for mask_filename in mask_files:
        print(f"正在处理: {mask_filename} ...")

        mask_path = os.path.join(MASK_DIR, mask_filename)
        base_name = os.path.splitext(mask_filename)[0]

        img_filename_guess1 = mask_filename.replace("_unet", "_1")
        img_filename_guess2 = mask_filename.replace("_unet", "")

        if os.path.exists(os.path.join(IMG_DIR, img_filename_guess1)):
            img_path = os.path.join(IMG_DIR, img_filename_guess1)
        elif os.path.exists(os.path.join(IMG_DIR, img_filename_guess2)):
            img_path = os.path.join(IMG_DIR, img_filename_guess2)
        else:
            img_path = os.path.join(IMG_DIR, mask_filename)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [警告] 无法读取掩码图 {mask_path}，跳过。")
            continue

        img_color = cv2.imread(img_path)
        if img_color is None:
            print(f"  [提示] 未找到对应的原图，将使用掩码图作为底图。")
            img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if mask.shape[:2] != img_color.shape[:2]:
            img_color = cv2.resize(img_color, (mask.shape[1], mask.shape[0]))

        merged_lines = extract_lines_from_mask(mask)

        np.random.seed(42)
        overlay = img_color.copy()
        overlay = cv2.addWeighted(overlay, 0.4, np.zeros_like(overlay), 0.6, 0)

        for idx, line in enumerate(merged_lines):
            x1, y1, x2, y2 = map(int, line)
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            cv2.line(overlay, (x1, y1), (x2, y2), color, 4)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            text = str(idx + 1)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(overlay, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
            cv2.putText(overlay, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        features = build_line_features(merged_lines)

        out_img_path = os.path.join(OUT_LINES_DIR, f"{base_name}_lines.jpg")
        out_json_path = os.path.join(OUT_JSONS_DIR, f"{base_name}.json")

        cv2.imwrite(out_img_path, overlay)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)

        print(f"  -> 完成！提取了 {len(merged_lines)} 条线段。")

    print("\n🎉 所有图像批量处理完毕！")


if __name__ == "__main__":
    main()
