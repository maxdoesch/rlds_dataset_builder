import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
import io

# Lade das Dataset
DATASET_PATH = "/home/maxi/tensorflow_datasets/"
dataset = tfds.load("lego_dataset", data_dir=DATASET_PATH, split='train', shuffle_files=False)

def check_png(episode, name):
    """Überprüft, ob die PNG-Daten gültig sind."""
    for step in episode['steps']:
        image = step['observation'][name]
        
        try:
            img = Image.open(io.BytesIO(image.numpy()))
            img.verify()  # Prüft auf Korrumpierung
            print(f"Valid PNG: {name}")
        except Exception as e:
            print(f"Corrupt PNG detected in {name}: {e}")

def check_gripper(episode, name):
    for step in episode['steps']:
        print(step['action'][6])

# Iteriere durch das Dataset und überprüfe die Bilder
for example in dataset.take(10):  # Nehme eine begrenzte Anzahl, um die Überprüfung zu testen
    check_gripper(example, "third_person_image")
    #check_png(example, "wrist_image")