import os

def check_yolo_pairs(images_dir, labels_dir, image_exts=(".jpg", ".jpeg", ".png")):
    images = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in image_exts]
    labels = [os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')]
    images_set, labels_set = set(images), set(labels)
    images_wo_labels = images_set - labels_set
    labels_wo_images = labels_set - images_set
    return list(images_wo_labels), list(labels_wo_images)

def main():
    splits = ['train', 'valid', 'test']
    base = '.'
    results = {}
    for split in splits:
        img_dir = os.path.join(base, split, 'images')
        lbl_dir = os.path.join(base, split, 'labels')
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            i_wo_l, l_wo_i = check_yolo_pairs(img_dir, lbl_dir)
            results[split] = {'images_wo_labels': i_wo_l, 'labels_wo_images': l_wo_i}
        else:
            results[split] = 'Missing images or labels directory'
    for split, res in results.items():
        print(f"\n=== {split.upper()} SPLIT ===")
        if isinstance(res, dict):
            print(f"Images without labels: {res['images_wo_labels']}")
            print(f"Labels without images: {res['labels_wo_images']}")
        else:
            print(res)

if __name__ == "__main__":
    main()
