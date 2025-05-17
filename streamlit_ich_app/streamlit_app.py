import streamlit as st 
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torch 


# ------------------ CONFIG ------------------ #

st.set_page_config(
    page_title="ICH Detection Assistant",
    page_icon="🧠",
    layout="wide"
    )


# --------------- SESSION SETUP --------------- #
if "page_selector" not in st.session_state:
    st.session_state.page_selector = "🏠 Home"

# ------------------ SIDEBAR ------------------ #
with st.sidebar:
    st.image("brain4.png", width=250)  
    st.markdown("## 📂 Navigation")
    selected_page = st.radio(
        "Go to:", 
        [" Home", " DCM Viewer", " Diagnosis", " Report", " About"
], 
        key="page_selector"
    )

  
    



# ------------------ FUNCTIONS ------------------ #

mean_img = [0.22363983, 0.18190407, 0.2523437 ]
std_img = [0.32451536, 0.2956294,  0.31335256]


def rescale_image(image, slope, intercept):
    return image * slope + intercept

def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image

def apply_window_policy(image):

    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)

    return image

def preprocess_dicom_image(dcm, size=512):
    img = dcm.pixel_array.astype(float)
    img = rescale_image(img, dcm.RescaleSlope, dcm.RescaleIntercept)
    img = apply_window_policy(img)
    img -= img.min((0,1))
    final_img = (img * 255).astype(np.uint8)  # For display

    img_pil = Image.fromarray(final_img)
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((size, size)),
        T.Normalize(mean = mean_img,
                    std = std_img)
    ])
    input_tensor = transform(img_pil).unsqueeze(0)  # Shape: [1, 3, H, W]

    return input_tensor, final_img


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# Load ResNeXt model architecture
def load_resnext_feature_extractor(model_path):
    model = torch.load('C:/Users/SarahAl2203.SARAHALLAPTOP/Desktop/streamlit_ich_app/resnext101_32x8d_wsl_checkpoint.pth', weights_only=False)

    for param in model.parameters():
        param.requires_grad = False

    #model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.module.fc = Identity()
    model.eval()
    return model

#extract emb from tensor

def extract_embedding_from_tensor(tensor, model):
    model.eval()
    with torch.no_grad():
        embedding = model(tensor)  # Shape: [1, 2048]
    return embedding.cpu().numpy().squeeze(0)




# -------------------- HOME -------------------- #

if selected_page == " Home":
    #SECTION 1 Welcoming 
    st.markdown("## 👋 Welcome to the ICH Detection Assistant")
    st.markdown("<p style='text-align: center;font-size: 20px;'>A clinically-informed AI tool to assist healthcare professionals in identifying intracranial hemorrhages from CT scans.</p>", unsafe_allow_html=True)
    st.markdown("""
<div style='text-align: center;font-size: 20px;'>

<p style="font-style: italic; color: #888;">
⚠️ <strong>Note:</strong> This tool is intended to <strong>support</strong>, not replace, clinical judgment.  
All diagnostic decisions must be made by qualified medical professionals in the context of the patient's full clinical picture.
</p>

<hr style="border: 1px solid #ccc;"/>

</div>
""", unsafe_allow_html=True)
    
    st.markdown("## 🩺 How to Use This Tool:")
    st.markdown("""
    <ol style="text-align: left; font-size: 20px;">
        <li><strong>Upload a patient's CT scan</strong></li>
        <li><strong>Run the AI model</strong></li>
        <li><strong>Download report for documentation</strong></li>
    </ol>
    <hr style="border: 1px solid #ccc;"/>
    """, unsafe_allow_html=True)
    
    
    #Section 2 Brain Anatomy
    st.markdown("## 🧠 ICH Subtypes", unsafe_allow_html=True)
    st.markdown("""
<style>
.ich-container {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    gap: 30px;
    margin-top: 20px;
    padding: 10px;
}

.ich-card {
    text-align: center;
    width: 130px;
}

.ich-card img {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #ccc;
    background-color: #f9f9f9;
    padding: 5px;
}

.ich-title {
    font-weight: bold;
    margin-top: 10px;
    font-size: 16px;
}
</style>

<div class="ich-container">
    <div class="ich-card">
        <img src="https://i.postimg.cc/sXyX6WDY/epidural.png" alt="Epidural">
        <div class="ich-title">Epidural</div>
    </div>
    <div class="ich-card">
        <img src="https://i.postimg.cc/LXcJKppq/subdural.png" alt="Subdural">
        <div class="ich-title">Subdural</div>
    </div>
    <div class="ich-card">
        <img src="https://i.postimg.cc/fL4Pw0K6/subarachnoid.png" alt="Subarachnoid">
        <div class="ich-title">Subarachnoid</div>
    </div>
    <div class="ich-card">
        <img src="https://i.postimg.cc/nrFdzZg4/intracerebral.png" alt="Intracerebral">
        <div class="ich-title">Intracerebral</div>
    </div>
    <div class="ich-card">
        <img src="https://i.postimg.cc/vB41ChNr/ivh.png" alt="Intraventricular">
        <div class="ich-title">Intraventricular</div>
    </div>
</div>
""", unsafe_allow_html=True)

    

    # This will later embed your Three.js or Spline viewer:
    # components.html(open("brain_viewer.html").read(), height=600)
 
# -------------------- DCM Viewer -------------------- #
elif selected_page == " DCM Viewer":
    st.markdown("## 🖥️ DCM Viewer")
    uploaded_files = st.file_uploader("Upload multiple DICOM slices", type=["dcm"], accept_multiple_files=True, key="multi")
    if uploaded_files:
        slices = []
        for file in uploaded_files:
            try:
                dcm = pydicom.dcmread(file)
                img = dcm.pixel_array.astype(float)
                img = rescale_image(img, dcm.RescaleSlope, dcm.RescaleIntercept)
                img = apply_window_policy(img)
                img -= img.min((0, 1))
                final_image = (img * 255).astype(np.uint8)
                slices.append({
                    "image": final_image,
                    "dcm": dcm,
                    "position": getattr(dcm, "ImagePositionPatient", [0])[2]
                    if hasattr(dcm, "ImagePositionPatient")
                    else dcm.get("InstanceNumber", 0)
                })
            except Exception as e:
                st.warning(f"Skipping a file due to error: {e}")
        slices = sorted(slices, key=lambda x: x["position"])
        if slices:
            idx = st.slider("Navigate slices", 0, len(slices) - 1, 0)
            selected = slices[idx]
            st.image(selected["image"], caption=f"Slice {idx + 1}", use_container_width=True)    
    
    

# -------------------- Diagnosis -------------------- #
elif selected_page == " Diagnosis":
    st.markdown("## 📤 Upload Scan")
    uploaded_file = st.file_uploader("Upload a single DICOM file", type=["dcm"], key="single")
    if uploaded_file:
        try:
            dcm = pydicom.dcmread(uploaded_file)
            input_tensor, final_image = preprocess_dicom_image(dcm)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(final_image, caption="Uploaded CT Slice", use_container_width=True)
            with col2:
                st.write({
                    "Patient ID": dcm.get("PatientID", "N/A"),
                    "Modality": dcm.get("Modality", "N/A"),
                    "Study Date": dcm.get("StudyDate", "N/A"),
                    "Slice Thickness": dcm.get("SliceThickness", "N/A"),
                    "Pixel Spacing": dcm.get("PixelSpacing", "N/A"),
                    "Image Size": f"{final_image.shape}"
                })
            with col3:
                selected_window = st.selectbox("Select Windowing Mode", ["Brain", "Subdural", "Bone"])
                def apply_selected_window(image, mode):
                    if mode == "Brain":
                        win_image = apply_window(image, 40, 80)
                        win_image = (win_image - 0) / 80
                    elif mode == "Subdural":
                        win_image = apply_window(image, 80, 200)
                        win_image = (win_image - (-20)) / 200
                    elif mode == "Bone":
                        win_image = apply_window(image, 40, 380)
                        win_image = (win_image - (-150)) / 380
                    else:
                        return None
                    return (win_image * 255).astype(np.uint8)
                windowed_output = apply_selected_window(
                    dcm.pixel_array.astype(float) * dcm.RescaleSlope + dcm.RescaleIntercept,
                    selected_window
                )
                if windowed_output is not None:
                    st.image(windowed_output, caption=f"{selected_window} Window View", use_container_width=True)
        except Exception as e:
            st.error(f"Error reading DICOM file: {e}")
    st.markdown("## Run ICH Detection")
    if uploaded_file:
        if st.button("Run Detection"):
            try:
                # Extract embedding from the uploaded tensor
                #embedding = extract_embedding_from_tensor(input_tensor.to(device), resnext_model)

                st.success("✅ Embedding extracted successfully!")
                #st.write("**Embedding shape:**", embedding.shape)

                # 🔜 Later: Pass embedding to LSTM and show prediction
                st.info("This is a mock step. Integration with LSTM coming next!")

            except Exception as e:
                st.error(f"Model inference failed: {e}")

                
                

    # Modeling section

    #model_path = 'C:/Users/SarahAl2203.SARAHALLAPTOP/Desktop/streamlit_ich_app/model_999_epoch2_fold6.bin'

    # Load ResNeXt once
    #resnext_model = load_resnext_feature_extractor(model_path)




# -------------------- Report -------------------- #
elif selected_page == " Report":
    st.markdown("## 📄 Downloadable Report")
    if st.button("Generate & Download Report"):
        with open("ich_report.txt", "w") as f:
            f.write("ICH Detection Report\n\nResult: No hemorrhage detected.")
        with open("ich_report.txt", "rb") as f:
            st.download_button("📥 Download Report", f, "ich_report.txt", mime="text/plain")



    
    
    
elif selected_page == " About":
    st.markdown("## 🔬 Abstract & Research Team")
    st.markdown("### Abstract")
    st.markdown("""
<div style='text-align: left;'>

<p>
Intracranial hemorrhage detection (ICH) is a serious medical condition that necessitates a prompt and exhaustive medical diagnosis. This project a multi-label ICH classification issue with six different types of hemorrhages, namely Epidural (EPD), Intraparenchymal (ITP), Intraventricular (ITV), Subarachnoid (SBC), Subdural (SBD), and Any. This project presents a hybrid deep learning approach that combines Convolutional Neural Networks (CNN) and Long Short Term Memory approaches (LSTM). Leveraging a ResNeXt-101 convolutional neural network for spatial feature extraction with a bidirectional LSTM network to capture temporal dependencies across slice sequences. Comprehensive preprocessing steps—including DICOM parsing, Hounsfield Unit calibration, multi-window image construction, and dynamic augmentation via Albumentations—were applied to enhance input consistency and generalization. Labels were reformatted into a multi-label binary matrix, enabling simultaneous classification of six hemorrhage types. The model achieved a private leaderboard log loss of 0.04604 on the RSNA 2019 ICH Detection Challenge, demonstrating strong generalization to unseen data. Despite challenges such as data imbalance and the absence of 3D context, the proposed system lays a foundation for explainable and scalable clinical AI. Future work will focus on integrating retrieval-augmented report generation and radiologist feedback mechanisms to further support real-world deployment.

</p>

<hr style="border: 1px solid #ccc;"/>

</div>
""", unsafe_allow_html=True)

    st.markdown("""
    <style>
        .author-container {
            display: flex;
            justify-content: center;
            gap: 80px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .author-card {
            text-align: center;
            font-family: 'Arial', sans-serif;
            padding: 10px;
        }
        .author-img {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #ddd;
            margin-bottom: 10px;
        }
        .author-name {
            font-size: 20px;
            margin: 5px 0;
        }
        .author-links a {
            text-decoration: none;
            margin: 0 10px;
            font-size: 20px;
            color: #333;
        }
        .author-links a:hover {
            color: #0072b1;
        }
    </style>

    <div class="author-container">
        <div class="author-card">
            <img src="" class="author-img" alt="Ahmad M">
            <p class="author-name" style="font-weight: bold;">Ahmad M Alqaisi</p>
            <p style="font-size: 14px; color: #555; margin-top: 20px; margin-bottom: 5px; line-height: 1.2;">
                Data Science and AI Graduate from Al Albayt University<br>
                Deep Learning, ML, Data Engineering<br>
                Certified by IBM & Alteryx
            </p>
            <div class="author-links">
                <a href="mailto:moksasbeh@gmail.com" target="_blank"><i class="fa-solid fa-envelope"></i></a>
                <a href="https://www.linkedin.com/in/mones-ksasbeh" target="_blank"><i class="fab fa-linkedin"></i></a>
            </div>
        </div>
        <div class="author-card">
            <img src="https://i.postimg.cc/CK12pb7s/IMG-0068-1.jpg" class="author-img" alt="Sarah Al">
            <p class="author-name" style="font-weight: bold;">Sarah K Almashagbeh</p>
            <p style="font-size: 14px; color: #555; margin-top: 20px; margin-bottom: 5px; line-height: 1.2;">
                Data Science and AI Graduate from Al Albayt University<br>
                Passionate About Medical Imaging, Data Science, Deep Learning<br> Aspiring AI researcher
            </p>
            <div class="author-links">
                <a href="mailto:am5294690@gmail.com" target="_blank"><i class="fa-solid fa-envelope"></i></a>
                <a href="https://www.linkedin.com/in/yazan-mansour-7644aa264" target="_blank"><i class="fab fa-linkedin"></i></a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

