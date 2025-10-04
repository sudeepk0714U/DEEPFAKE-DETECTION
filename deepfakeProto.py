import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN


# --- Build Meso4 Model ---
def build_meso():
    image_dimensions = {'height': 256, 'width': 256, 'channels': 3}
    x = Input(shape=(image_dimensions['height'], image_dimensions['width'], image_dimensions['channels']))
    x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

    x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=x, outputs=y)
    model.load_weights('./weights/Meso4_DF.h5')
    return model


# --- Load model and detector ---
@st.cache_resource
def load_all():
    return build_meso(), MTCNN()


model, detector = load_all()

# --- Streamlit UI ---
st.title("🎥 Deepfake Detection using Meso4 (with Face Cropping)")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = open("temp_video.mp4", "wb")
    tfile.write(uploaded_file.read())
    st.video("temp_video.mp4")

    if st.button("Analyze Video"):
        cap = cv2.VideoCapture("temp_video.mp4")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, frame_count // 30)

        fake_scores = []
        progress = st.progress(0)
        total = frame_count // frame_skip

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if int(cap.get(1)) % frame_skip != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            if faces:
                x, y, w, h = faces[0]['box']
                face = rgb[y:y + h, x:x + w]
                face = cv2.resize(face, (256, 256))
                face = img_to_array(face) / 255.0
                face = np.expand_dims(face, axis=0)
                pred = model.predict(face, verbose=0)[0][0]
                fake_scores.append(pred)

            progress.progress(int((i + 1) / total * 100))

        cap.release()

        if len(fake_scores) == 0:
            st.error("No face detected in video.")
        else:
            prob = np.median(fake_scores)
            st.subheader(f"🧠 Deepfake Probability: {prob:.2f}")
            if prob > 0.6:
                st.success("⚠️ Likely Real Video")
            else:
                st.error("✅ Likely Deepfake Video")
