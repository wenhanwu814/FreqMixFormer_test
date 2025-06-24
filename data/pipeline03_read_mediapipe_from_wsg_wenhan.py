import os
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cv2
import json

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

PERSON_ID_CHANNEL = 95
KEYPOINT_ID_CHANNEL = 96
SCORE_CHANNEL = 66
Z_CHANNEL = 97

# 33点的逆向映射
def get_mediapipe_keypoint_mapping():
    return {i: f"pose_{i}" for i in range(33)}

def extract_mediapipe_from_wsg(wsg_array):
    if wsg_array is None:
        return None
    height, width = wsg_array.shape[:2]
    keypoint_mapping = get_mediapipe_keypoint_mapping()
    person_channel = wsg_array[:, :, PERSON_ID_CHANNEL]
    keypoint_channel = wsg_array[:, :, KEYPOINT_ID_CHANNEL]
    score_channel = wsg_array[:, :, SCORE_CHANNEL]
    z_channel = wsg_array[:, :, Z_CHANNEL]
    valid_mask = (person_channel != -1) & (keypoint_channel != -1) & (score_channel != -1)
    valid_coords = np.where(valid_mask)
    if len(valid_coords[0]) == 0:
        return None
    persons_raw_data = {}
    for i in range(len(valid_coords[0])):
        y, x = valid_coords[0][i], valid_coords[1][i]
        person_id = int(person_channel[y, x])
        keypoint_id = int(keypoint_channel[y, x])
        score = float(score_channel[y, x])
        z = float(z_channel[y, x])
        if person_id < 0 or keypoint_id not in keypoint_mapping:
            continue
        if person_id not in persons_raw_data:
            persons_raw_data[person_id] = {}
        # 归一化坐标
        normalized_x = x / width
        normalized_y = y / height
        persons_raw_data[person_id][keypoint_id] = (normalized_x, normalized_y, z, score)
    if not persons_raw_data:
        return None
    mediapipe_data = {}
    for person_id, raw_data in persons_raw_data.items():
        keypoints = np.full((33, 4), -1.0, dtype=np.float32)
        for kp_idx, (x, y, z, score) in raw_data.items():
            if 0 <= kp_idx < 33:
                keypoints[kp_idx, 0] = x
                keypoints[kp_idx, 1] = y
                keypoints[kp_idx, 2] = z
                keypoints[kp_idx, 3] = score
        mediapipe_data[f'person_{person_id}'] = keypoints
    return mediapipe_data

def process_single_wsg_file(args):
    wsg_path, output_dir, modes = args
    try:
        base_name = os.path.splitext(os.path.basename(wsg_path))[0]
        wsg_array = np.load(wsg_path, allow_pickle=True)['wsg_data']
        mediapipe_data = extract_mediapipe_from_wsg(wsg_array)
        if mediapipe_data is None:
            print(f"  跳过 {base_name}: 未找到mediapipe数据")
            return False
        height, width = wsg_array.shape[:2]
        # 获取原始RGB图像
        original_rgb = wsg_array[:, :, 0:3].astype(np.uint8)
        output_image = original_rgb.copy()
        # 可视化
        if 'vis' in modes or 'vis_overlay' in modes:
            for person_id, keypoints in mediapipe_data.items():
                color = (0, 255, 0)
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if (keypoints[start_idx, 3] > 0.3 and keypoints[end_idx, 3] > 0.3):
                        start_point = (int(keypoints[start_idx, 0] * width), int(keypoints[start_idx, 1] * height))
                        end_point = (int(keypoints[end_idx, 0] * width), int(keypoints[end_idx, 1] * height))
                        cv2.line(output_image, start_point, end_point, color, 2)
                for x, y, z, v in keypoints:
                    if v > 0.3:
                        cv2.circle(output_image, (int(x * width), int(y * height)), 4, color, -1)
                        cv2.circle(output_image, (int(x * width), int(y * height)), 4, (255, 255, 255), 1)
        if 'vis' in modes:
            vis_path = os.path.join(output_dir, f"export_mediapipe_from_wsg_{base_name}_vis.png")
            cv2.imwrite(vis_path, output_image)
            print(f"    保存骨骼可视化: {vis_path}")
        if 'vis_overlay' in modes:
            original_pil = Image.fromarray(original_rgb)
            overlay_img = Image.blend(original_pil.convert('RGBA'), Image.fromarray(output_image).convert('RGBA'), alpha=0.5)
            overlay_path = os.path.join(output_dir, f"export_mediapipe_from_wsg_{base_name}_overlay.png")
            overlay_img.save(overlay_path)
            print(f"    保存叠加可视化: {overlay_path}")
        if 'save' in modes:
            npz_path = os.path.join(output_dir, f"export_mediapipe_from_wsg_{base_name}.npz")
            np.savez_compressed(npz_path, **mediapipe_data)
            print(f"    保存NPZ数据: {npz_path}")
        if 'json' in modes:
            json_path = os.path.join(output_dir, f"export_mediapipe_from_wsg_{base_name}.json")
            with open(json_path, 'w') as f:
                json.dump({k: v.tolist() for k, v in mediapipe_data.items()}, f)
            print(f"    保存JSON数据: {json_path}")
        return True
    except Exception as e:
        print(f"  处理文件失败 {wsg_path}: {str(e)}")
        return False

def process_wsg_input(input_path, output_path, modes, max_workers=4):
    if os.path.isfile(input_path) and input_path.endswith('.wsg'):
        if output_path is None:
            output_path = os.path.dirname(input_path)
        os.makedirs(output_path, exist_ok=True)
        print(f"处理单个WSG文件: {input_path}")
        print(f"输出目录: {output_path}")
        success = process_single_wsg_file((input_path, output_path, modes))
        print(f"处理完成! 成功: {'是' if success else '否'}")
    elif os.path.isdir(input_path):
        if output_path is None:
            parent_dir = os.path.dirname(input_path.rstrip('/'))
            folder_name = os.path.basename(input_path.rstrip('/'))
            output_path = os.path.join(parent_dir, f"export_mediapipe_from_{folder_name}")
        os.makedirs(output_path, exist_ok=True)
        wsg_files = glob.glob(os.path.join(input_path, "*.wsg"))
        wsg_files.sort()
        if not wsg_files:
            print(f"在目录 {input_path} 中没有找到WSG文件")
            return
        print(f"处理WSG文件夹: {input_path}")
        print(f"输出目录: {output_path}")
        print(f"找到 {len(wsg_files)} 个WSG文件")
        tasks = [(wsg_path, output_path, modes) for wsg_path in wsg_files]
        success_count = 0
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_wsg_file, task) for task in tasks]
                for future in tqdm(futures, desc="提取mediapipe"):
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        print(f"任务失败: {str(e)}")
        else:
            for task in tqdm(tasks, desc="提取mediapipe"):
                if process_single_wsg_file(task):
                    success_count += 1
        print(f"处理完成! 成功: {success_count}/{len(wsg_files)}")
    else:
        print(f"无效的输入路径: {input_path}")

def main():
    # 自动遍历所有视频文件夹
    input_base = "../output/test_dataset_results"
    modes = ['vis', 'vis_overlay', 'save', 'json']  # 可选: 'vis', 'vis_overlay', 'save', 'json'
    max_workers = mp.cpu_count() // 2
    print("=" * 50)
    print("WSG mediapipe提取和可视化工具 (批量模式)")
    print("=" * 50)
    folders = [f for f in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, f))]
    for folder in folders:
        wsg_dir = os.path.join(input_base, folder, "wsg")
        if os.path.isdir(wsg_dir):
            print(f"\n>>> 处理: {wsg_dir}")
            process_wsg_input(wsg_dir, None, modes, max_workers)
    print("\n✅ 所有提取任务完成!")

if __name__ == '__main__':
    main()
