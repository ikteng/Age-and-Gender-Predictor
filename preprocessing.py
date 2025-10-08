import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

IMG_SIZE = 128

class FolderImageDataLoader:
    def __init__(self, folder_path, image_size=(IMG_SIZE, IMG_SIZE), save_path="processed_data.npz"):
        self.folder_path = folder_path
        self.image_size = image_size
        self.save_path = save_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    def load_all(self):
        images, ages, genders = [], [], []
        skipped_invalid = 0
        skipped_format = 0

        for file_name in tqdm(self.file_list, desc="Loading images"):
            parts = file_name.split("_")
            if len(parts) < 2:
                skipped_format += 1
                continue

            try:
                age = int(parts[0])
                gender = int(parts[1])
            except ValueError:
                skipped_format += 1
                continue

            # --- Skip invalid gender labels (not 0 or 1) ---
            if gender not in [0, 1]:
                skipped_invalid += 1
                continue

            path = os.path.join(self.folder_path, file_name)
            try:
                img = Image.open(path).convert("RGB").resize(self.image_size)
                images.append(np.array(img))
                ages.append(age)
                genders.append(gender)
            except Exception as e:
                print(f"⚠️ Skipping {file_name}: {e}")
                continue

        X = np.array(images)
        y_age = np.array(ages)
        y_gender = np.array(genders)

        print(f"\n✅ Loaded {len(X)} valid images.")
        print(f"🚫 Skipped {skipped_invalid} with invalid gender, {skipped_format} invalid filenames.")
        print(f"Class counts → Male: {(y_gender == 0).sum()}, Female: {(y_gender == 1).sum()}")

        # Save processed data
        np.savez_compressed(self.save_path, X=X, y_age=y_age, y_gender=y_gender)
        print(f"\n💾 Saved cleaned dataset to {self.save_path}")

        return X, y_age, y_gender


def show_samples_by_gender(X, y_gender, samples_per_gender=5):
    genders = {0: "Male", 1: "Female"}
    plt.figure(figsize=(samples_per_gender * 2, 4))

    for gender_label in [0, 1]:
        idxs = np.where(y_gender == gender_label)[0]
        if len(idxs) == 0:
            continue

        chosen_idxs = np.random.choice(idxs, min(samples_per_gender, len(idxs)), replace=False)
        for i, idx in enumerate(chosen_idxs):
            plt.subplot(2, samples_per_gender, gender_label * samples_per_gender + i + 1)
            plt.imshow(X[idx])
            plt.axis("off")
            if i == 0:
                plt.ylabel(genders[gender_label], fontsize=12)

    plt.suptitle("Sample Images by Gender", fontsize=16)
    plt.show()


if __name__ == "__main__":
    folder_path = "crop_part1"
    image_size = (IMG_SIZE, IMG_SIZE)
    npz_path = f"utkface_processed_{IMG_SIZE}.npz"

    data_loader = FolderImageDataLoader(folder_path, image_size=image_size, save_path=npz_path)
    X_all, y_age_all, y_gender_all = data_loader.load_all()

    # Display 5 samples per gender
    show_samples_by_gender(X_all, y_gender_all, samples_per_gender=5)
