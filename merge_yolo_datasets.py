import os
import shutil
import yaml

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

def copy_and_remap_labels(src_labels, dst_labels, id_map):
    os.makedirs(dst_labels, exist_ok=True)
    for fname in os.listdir(src_labels):
        src_file = os.path.join(src_labels, fname)
        dst_file = os.path.join(dst_labels, fname)
        with open(src_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id in id_map:
                parts[0] = str(id_map[cls_id])
                new_lines.append(" ".join(parts))
        with open(dst_file, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))

def copy_images(src_images, dst_images):
    os.makedirs(dst_images, exist_ok=True)
    for fname in os.listdir(src_images):
        src_file = os.path.join(src_images, fname)
        dst_file = os.path.join(dst_images, fname)
        if os.path.exists(dst_file):
            name, ext = os.path.splitext(fname)
            dst_file = os.path.join(dst_images, f"{name}_2{ext}")
        shutil.copy2(src_file, dst_file)

def merge_yolo_datasets(dataset1, dataset2, output_dir="merged_dataset"):
    os.makedirs(output_dir, exist_ok=True)
    subsets = ["train", "valid", "test"]

    # 读取两个 yaml
    yaml1 = load_yaml(os.path.join(dataset1, "data.yaml"))
    yaml2 = load_yaml(os.path.join(dataset2, "data.yaml"))

    # 合并类别列表
    all_names = []
    for n in yaml1["names"] + yaml2["names"]:
        if n not in all_names:
            all_names.append(n)

    # 建立类别映射
    id_map_1 = {i: all_names.index(name) for i, name in enumerate(yaml1["names"])}
    id_map_2 = {i: all_names.index(name) for i, name in enumerate(yaml2["names"])}

    # 合并图像与标签并重映射类别ID
    for subset in subsets:
        print(f"处理 {subset} ...")
        for sub in ["images", "labels"]:
            src1 = os.path.join(dataset1, subset, sub)
            src2 = os.path.join(dataset2, subset, sub)
            dst = os.path.join(output_dir, subset, sub)
            os.makedirs(dst, exist_ok=True)
            if sub == "images":
                if os.path.exists(src1): copy_images(src1, dst)
                if os.path.exists(src2): copy_images(src2, dst)
            else:  # labels
                if os.path.exists(src1): copy_and_remap_labels(src1, dst, id_map_1)
                if os.path.exists(src2): copy_and_remap_labels(src2, dst, id_map_2)

    # 写新的 YAML
    merged_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": all_names,
        "nc": len(all_names)
    }
    save_yaml(os.path.join(output_dir, "data.yaml"), merged_yaml)
    print(f"数据集合并完成！新类别数量: {len(all_names)}")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    dataset1 = "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\data_full"
    dataset2 = "D:\\Codefield\\MyPython\\DeepLearning\\Projects\\FinalProject\\FP_test\\data_new_add"
    merge_yolo_datasets(dataset1, dataset2)


