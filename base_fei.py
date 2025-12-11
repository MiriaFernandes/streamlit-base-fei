import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from pymongo import MongoClient
from PIL import Image
import io
import base64

st.markdown(
"""
<style>
body {
    background-color: #ADD8E6;
}
.css-1d391kg {
    background-color: #87CEEB;
}
.stButton>button {
    background-color: #1E90FF;
    color: white;
}
</style>
""",
unsafe_allow_html=True
)
# --------------------------
# Configuração do MongoDB
# --------------------------
MONGO_URI = "mongodb+srv://miriafernandes_db_user:InlLM3UlVUSUnmHq@cluster0.rftdpge.mongodb.net/?appName=Cluster0"
DB_NAME = "atividade14"
COLLECTION_NAME = "imagens"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# --------------------------
# Funções auxiliares
# --------------------------
def imagem_para_base64(imagem_bytes):
    return base64.b64encode(imagem_bytes).decode("utf-8")

def base64_para_imagem(image_base64):
    return Image.open(io.BytesIO(base64.b64decode(image_base64)))

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def extrair_embedding(imagem_bytes):
    image = cv2.imdecode(np.frombuffer(imagem_bytes, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face.process(image_rgb)
    if not results.detections:
        return None
    h, w, _ = image_rgb.shape
    bbox = results.detections[0].location_data.relative_bounding_box
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = x1 + int(bbox.width * w)
    y2 = y1 + int(bbox.height * h)
    face_crop = image_rgb[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
    face_resized = cv2.resize(face_crop, (64, 64))
    embedding = face_resized.flatten() / 255.0
    return embedding

def salvar_imagem_no_mongo(img):
    img_bytes = img.getvalue()
    embedding = extrair_embedding(img_bytes)
    if embedding is None:
        return False
    documento = {
        "nome_arquivo": img.name,
        "imagem_base64": imagem_para_base64(img_bytes),
        "embedding": embedding.tolist()
    }
    collection.insert_one(documento)
    return True

def encontrar_similaridades(embedding_usuario):
    documentos = list(collection.find())
    if not documentos:
        return None, None
    resultados = []
    for doc in documentos:
        embedding_banco = np.array(doc["embedding"])
        distancia = np.linalg.norm(embedding_usuario - embedding_banco)
        resultados.append({
            "nome": doc["nome_arquivo"],
            "distancia": distancia,
            "imagem_base64": doc["imagem_base64"]
        })
    mais_parecido = min(resultados, key=lambda x: x["distancia"])
    mais_diferente = max(resultados, key=lambda x: x["distancia"])
    return mais_parecido, mais_diferente

# --------------------------
# Interface Streamlit
# --------------------------
st.title("Banco de Imagens FEI - MongoDB")

menu = st.sidebar.selectbox(
    "Opções",
    ["Adicionar fotos ao MongoDB", "Comparar com minha fotografia"]
)

if menu == "Adicionar fotos ao MongoDB":
    st.header("Carregar imagens no MongoDB")
    imagens = st.file_uploader("Escolha arquivos para guardar no banco", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if st.button("Gravar no MongoDB"):
        if imagens:
            total_salvas = sum(salvar_imagem_no_mongo(img) for img in imagens)
            st.success(f"{total_salvas} imagens foram armazenadas com êxito no MongoDB!")
        else:
            st.warning("Nenhuma imagem foi selecionada.")

elif menu == "Comparar com minha fotografia":
    st.header("Identificar a imagem mais semelhante e a mais distinta")
    minha_imagem = st.file_uploader("Envie uma fotografia", type=["jpg","png","jpeg"])
    if minha_imagem:
        img_bytes = minha_imagem.getvalue()
        st.image(Image.open(io.BytesIO(img_bytes)), caption="Sua fotografia", width=250)
        embedding_usuario = extrair_embedding(img_bytes)
        if embedding_usuario is None:
            st.error("Nenhum rosto encontrado na imagem enviada!")
        else:
            mais_parecido, mais_diferente = encontrar_similaridades(embedding_usuario)
            if not mais_parecido or not mais_diferente:
                st.warning("O banco ainda não possui imagens.")
            else:
                st.subheader("✅ Maior semelhança")
                st.image(base64_para_imagem(mais_parecido["imagem_base64"]), width=200)
                st.write(f"Nome: {mais_parecido['nome']}")
                st.write(f"Distância: {mais_parecido['distancia']:.4f}")

                st.subheader("❌ Maior diferença")
                st.image(base64_para_imagem(mais_diferente["imagem_base64"]), width=200)
                st.write(f"Nome: {mais_diferente['nome']}")
                st.write(f"Distância: {mais_diferente['distancia']:.4f}")