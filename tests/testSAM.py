import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time
import csv
import os

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ======================= Utils =======================

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# ======================= Configuration =======================

vit_types = ['vit_b', 'vit_l', 'vit_h']
points_per_side_list = [8, 16, 32]
device = "cpu"  
torch.set_num_threads(os.cpu_count())
print(f"PyTorch va utiliser {torch.get_num_threads()} threads CPU")


MODEL_PATHS = {
    'vit_b': 'sam_vit_b_01ec64.pth',
    'vit_l': 'sam_vit_l_0b3195.pth',
    'vit_h': 'sam_vit_h_4b8939.pth'
}

image_path = 'images/interieur2.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ======================= Benchmark Loop =======================

results = []
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


for vit in vit_types:
    print(f"\n=== 🔍 Chargement modèle {vit} ===")
    start_load = time.time()
    sam = sam_model_registry[vit](checkpoint=MODEL_PATHS[vit])
    sam.to(device)
    load_time = time.time() - start_load
    print(f"Temps de chargement : {load_time:.2f} sec")

    for pps in points_per_side_list:
        print(f"--- ⚙️ Génération avec points_per_side = {pps} ---")
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=pps
        )

        start = time.time()
        masks = mask_generator.generate(image)
        duration = time.time() - start

        print(f"Temps de génération : {duration:.2f} sec - {len(masks)} masques")

        #Nom du fichier image généré
        output_img_name = f"mask_{vit}_{pps}.png"
        output_img_path = os.path.join(output_dir, output_img_name)

        # Enregistrement des résultats
        results.append({
            "vit_type": vit,
            "points_per_side": pps,
            "load_time_s": round(load_time, 2),
            "generate_time_s": round(duration, 2),
            "num_masks": len(masks),
            "image_path":output_img_path
        })

        # === Générer l'image avec masques superposés et la sauvegarder ===
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        show_anns(masks)
        plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()


# ======================= Sauvegarde CSV =======================

csv_path = "benchmark_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Benchmark terminé. Résultats enregistrés dans : {csv_path}")