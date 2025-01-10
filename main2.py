from transformers import AutoModel, AutoFeatureExtractor
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from deepface import DeepFace

# === 1. Modelni yuklash ===
def load_model():
    model_name = "jayanta/vit-base-patch16-224-in21k-face-recognition"
    model = AutoModel.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

# === 2. Yuz embeddinglarini olish ===
def get_face_embedding(model, feature_extractor, image):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = feature_extractor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        return embeddings
    except Exception as e:
        print(f"Xatolik yuz berdi: {e}")
        return None

# === 3. Jinsni aniqlash ===
def detect_gender(image):
    try:
        analysis = DeepFace.analyze(img_path=image, actions=['gender'])
        return analysis['gender']
    except Exception as e:
        print(f"Jinsni aniqlashda xatolik yuz berdi: {e}")
        return "Unknown"

# === 4. Kameradan real vaqt rejimida yuzlarni aniqlash ===
def detect_and_compare_faces(model, feature_extractor, reference_embeddings, reference_names, font_path):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            embedding = get_face_embedding(model, feature_extractor, face)
            if embedding is not None:
                similarities = [calculate_similarity(embedding, ref_emb) for ref_emb in reference_embeddings]
                max_similarity = max(similarities)
                best_match_index = similarities.index(max_similarity)
                matched_name = reference_names[best_match_index] if max_similarity > 0.7 else "Not Recognized"

                # Konsolda natijalarni chiqarish
                print("O'xshashlik natijalari:", similarities)
                print("Eng yuqori o'xshashlik:", max_similarity)
                print("Tanlangan ism:", matched_name)

                # Yuzni ko'rsatish
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame_pil)
                font = ImageFont.truetype(font_path, 20)
                draw.text((x, y - 30), f"{matched_name}", font=font, fill=(255, 255, 255))
                frame = np.array(frame_pil)

        cv2.imshow("Real-Time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# === 5. Kosinus o'xshashligini hisoblash ===
def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# === 6. Asosiy kodni ishga tushirish ===
if __name__ == "__main__":
    model, feature_extractor = load_model()

    reference_images = [
        r"C:\Users\13\Desktop\face_recog\Camera Roll\WIN_20250110_15_56_44_Pro.jpg",
        r"C:\Users\13\Desktop\face_recog\Camera Roll\WIN_20250110_15_57_58_Pro.jpg",
        r"C:\Users\13\Desktop\face_recog\Camera Roll\WIN_20250110_15_58_15_Pro.jpg",
        r"C:\Users\13\Desktop\face_recog\Camera Roll\WIN_20250110_15_58_29_Pro.jpg",
    ]
    reference_names = ["우미다","이재익", "김영호", "김성민"]

    reference_embeddings = []
    for image_path in reference_images:
        image = cv2.imread(image_path)
        if image is not None:
            embedding = get_face_embedding(model, feature_extractor, image)
            if embedding is not None:
                reference_embeddings.append(embedding)
        else:
            print(f"Tasvir ochib bo'lmadi: {image_path}")

    if reference_embeddings:
        detect_and_compare_faces(model, feature_extractor, reference_embeddings, reference_names, font_path="malgun.ttf")
    else:
        print("Referens embeddinglar topilmadi.")
