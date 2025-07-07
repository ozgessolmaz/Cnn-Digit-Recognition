import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os

MODEL_PATH = "odev_model.keras"
IMAGE_PATH = "/Users/ozgesolmaz/Desktop/Gyk/cnn/pixilart-drawing.png"
INVERT_COLORS = False  
OUTPUT_DIR = "tahmin_sonuclari"

os.makedirs(OUTPUT_DIR, exist_ok=True) #Klasör oluşturma

model = tf.keras.models.load_model(MODEL_PATH)

img = Image.open(IMAGE_PATH).convert("L")
img = img.resize((28, 28))
if INVERT_COLORS:
    img = ImageOps.invert(img)

img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 28, 28, 1) # Normalize ve reshape işlemi

predictions = model.predict(img_array)
predicted_digit = np.argmax(predictions)
confidence_score = predictions[0][predicted_digit] * 100 # Tahminin güven skoru

print(f"Tahmin edilen rakam: {predicted_digit}")
print(f"Güven skoru: %{confidence_score:.2f}")
print("Tüm tahmin olasılıkları:", predictions)

plt.figure(figsize=(2, 2))
plt.imshow(img_array.reshape(28, 28), cmap='gray')  
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tahmin_sonucu.png"), bbox_inches='tight', pad_inches=0)
plt.close()


plt.figure(figsize=(8, 4))
bars = plt.bar(range(10), predictions[0], color="skyblue")
bars[predicted_digit].set_color("orange")
plt.title("0-9 Arası Tahmin Olasılıkları")
plt.xlabel("Rakam")
plt.ylabel("Olasılık")
plt.xticks(range(10))
plt.ylim(0, 1.0)
for i, p in enumerate(predictions[0]):
    plt.text(i, p + 0.02, f"%{p*100:.1f}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tahmin_grafiği.png"))
plt.close()

print("📊 Tahmin görseli ve grafik başarıyla oluşturuldu!")
