import streamlit as st 
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torch 

st.set_page_config(
    page_title="ICH Detection Assistant",
    page_icon="🧠",
    layout="wide"
    )
st.title("🧠 ICH Detection Assistant")




#ALL FUNCTIONS HERE

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




#########################################################333

#MAIN APP LOGIC
with st.sidebar:
    st.markdown("## 📂 Navigation")

    selected_section = st.radio("Go to Section:", [
    "Brain Anatomy",
    "Upload Scan",
    "Run Detection",
    "View Report",
    "Author's"
], index=0)
    st.markdown("---")
    st.markdown("""
    <h4 style="text-align: left;">🩺 How to Use This Tool:</h4>
    <ol style="text-align: left; font-size: 14px;">
        <li><strong>Upload</strong> a patient's CT scan (DICOM format — single slice or complete series)</li>
        <li><strong>Adjust the viewing window</strong> to highlight relevant structures (Brain, Subdural, Bone)</li>
        <li><strong>Run the AI model</strong> to assist in detecting potential hemorrhagic findings</li>
        <li><strong>Download a detailed report</strong> for documentation or further clinical review</li>
    </ol>
    """, unsafe_allow_html=True)





#SECTION 1 Welcoming 

st.markdown("""
<div style='text-align: center;'>
<h1>Welcome,</h1>
To a clinically-informed AI tool designed to support healthcare professionals in the <strong>early and accurate detection of ICH subtypes</strong> from non-contrast CT brain scans.

<p style="font-style: italic; color: #888;">
⚠️ <strong>Note:</strong> This tool is intended to <strong>support</strong>, not replace, clinical judgment.  
All diagnostic decisions must be made by qualified medical professionals in the context of the patient's full clinical picture.
</p>

<hr style="border: 1px solid #ccc;"/>

</div>
""", unsafe_allow_html=True)

#Section 2 Brain Anatomy
if selected_section == "Brain Anatomy":
    st.markdown("##  Brain Anatomy")
    st.markdown("_Explore hemorrhage-prone brain regions interactively._")

    st.markdown('<hr style="border: 1px solid #ccc;"/>', unsafe_allow_html=True)


# This will later embed your Three.js or Spline viewer:
# components.html(open("brain_viewer.html").read(), height=600)


#Section 3
if selected_section == "Upload Scan":
    st.markdown("## 📤 Upload Scan")
    st.markdown("""
    <p style='text-align: Left; font-size: 1.25em; color: #bbb;'>
    Upload CT scans to detect signs of Intracranial Hemorrhage using AI.
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🖼️ Single DICOM", "📂 Multiple DICOM Slices"])

    # ---------------- TAB 1: SINGLE DICOM ---------------- #
    with tab1:
        uploaded_file = st.file_uploader("Upload a single DICOM file", type=["dcm"], key="single")

        if uploaded_file:
            try:
                dcm = pydicom.dcmread(uploaded_file)
                input_tensor, final_image = preprocess_dicom_image(dcm)

                st.session_state["final_image"] = final_image
                st.session_state["uploaded_file"] = uploaded_file

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### 🖼️ CT Slice")
                    st.image(final_image, caption="Uploaded CT Slice", use_container_width=True)

                with col2:
                    st.markdown("### 📄 DICOM Metadata")
                    st.write({
                        "Patient ID": dcm.get("PatientID", "N/A"),
                        "Modality": dcm.get("Modality", "N/A"),
                        "Study Date": dcm.get("StudyDate", "N/A"),
                        "Slice Thickness": dcm.get("SliceThickness", "N/A"),
                        "Pixel Spacing": dcm.get("PixelSpacing", "N/A"),
                        "Image Size": f"{final_image.shape}"
                    })

                with col3:
                    st.markdown("### 🔍 Select Windowing Mode")
                    selected_window = st.selectbox("", ["Brain", "Subdural", "Bone"])

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
                    else:
                        st.warning("Failed to apply windowing.")

            except Exception as e:
                st.error(f"Error reading DICOM file: {e}")

    # ---------------- TAB 2: MULTI-SLICE DICOM ---------------- #
    with tab2:
        uploaded_files = st.file_uploader("Upload multiple DICOM slices", type=["dcm"], accept_multiple_files=True, key="multi")

        if uploaded_files:
            slices = []

            for file in uploaded_files:
                try:
                    dcm = pydicom.dcmread(file)
                    img = dcm.pixel_array.astype(float)
                    img = rescale_image(img, dcm.RescaleSlope, dcm.RescaleIntercept)
                    img = apply_window_policy(img)
                    img -= img.min((0,1))
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

            # Sort slices by Z-position
            slices = sorted(slices, key=lambda x: x["position"])

            if slices:
                idx = st.slider("Navigate slices", 0, len(slices) - 1, 0)
                selected = slices[idx]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### 🖼️ Slice {idx + 1}")
                    st.image(selected["image"], caption=f"Slice {idx + 1}", use_container_width=True)

                with col2:
                    dcm = selected["dcm"]
                    st.markdown("### 📄 DICOM Metadata")
                    st.write({
                        "Patient ID": dcm.get("PatientID", "N/A"),
                        "Modality": dcm.get("Modality", "N/A"),
                        "Study Date": dcm.get("StudyDate", "N/A"),
                        "Slice Thickness": dcm.get("SliceThickness", "N/A"),
                        "Pixel Spacing": dcm.get("PixelSpacing", "N/A"),
                        "Image Size": f"{selected['image'].shape}"
                    })

    
# Modeling section

#model_path = 'C:/Users/SarahAl2203.SARAHALLAPTOP/Desktop/streamlit_ich_app/model_999_epoch2_fold6.bin'

# Load ResNeXt once
#resnext_model = load_resnext_feature_extractor(model_path)

st.markdown("## 🧪 Run ICH Detection")

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


st.markdown('<hr style="border: 1px solid #ccc;"/>', unsafe_allow_html=True)
st.markdown("## 📄 Downloadable Report")


if st.button("Generate & Download Report"):
        # Example: mock PDF generation
        with open("ich_report.txt", "w") as f:
            f.write("ICH Detection Report\n\nResult: No hemorrhage detected.")  # Replace with real results
        with open("ich_report.txt", "rb") as f:
            st.download_button("📥 Download Report", f, "ich_report.txt", mime="text/plain")

if selected_section == "Author's":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>👤 Ahmad</h2>", unsafe_allow_html=True)

    st.image("ahmad_profile.png", width=150, caption="", use_container_width=False)

    st.markdown("""
    <div style='text-align: center; font-size: 16px; line-height: 1.6;'>
        Data Science & AI Enthusiast from Jordan 🇯🇴<br>
        Passionate about Deep Learning, Medical AI, and Clinical Applications<br>
        Experienced with PyTorch, Streamlit, DICOM, and AI-based healthcare tools<br>
        <br>
        Certified by IBM & Coursera<br><br>

        <a href='mailto:you@example.com'>📧</a>
        &nbsp;&nbsp;
        <a href='https://www.linkedin.com/in/your-profile' target='_blank'>🔗 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
