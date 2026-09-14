import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from src.model_architecture import Model

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_PATH = 'models/best_model.pth'

@st.cache_resource
def load_model():
    model = Model(num_classes=6)
    state = torch.load(MODEL_PATH, map_location='cpu')
    # Ako je sačuvan kao {'state_dict': ...} ili {'model_state_dict': ...}
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.set_page_config(page_title="Klasifikacija otpada", page_icon="♻️")
st.title("♻️ Klasifikacija otpada")
st.write("Uploaduj sliku da bi model predvideo vrstu otpada.")

uploaded_file = st.file_uploader("Izaberi sliku...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploadovana slika', use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs[0], dim=0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

    st.success(f"**Predviđena klasa:** {CLASSES[pred_class]}")
    st.metric("Pouzdanost", f"{confidence:.2%}")

    st.subheader("Verovatnoće po klasama")
    for cls, prob in zip(CLASSES, probs):
        st.progress(float(prob), text=f"{cls}: {prob:.2%}")