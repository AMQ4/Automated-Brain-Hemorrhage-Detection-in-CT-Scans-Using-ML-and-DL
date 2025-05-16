import streamlit as st 
if not st.session_state.get("started"):
    st.warning("Please start from the Welcome page first.")
    st.stop()
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#import albumentations as A
#from albumentations.pytorch import ToTensorV2


st.set_page_config(
    page_title="ICH Detection Assistant",
    page_icon="🧠",
    layout="centered"
    )
st.title("🧠 ICH Detection Assistant")

#ALL FUNCTIONS HERE
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

'''class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x'''

#########################################################333
# Title
st.markdown("    <p style='text-align: Left; font-size: 1.25em; color: #bbb;'>Upload CT scans to detect signs of<br>Intracranial Hemorrhage using AI.</p>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.write("Our AI model analyzes CT scan slices to identify signs of hemorrhage.")
    st.write("It uses deep learning and computer vision techniques to highlight areas of concern.")


     
tab1, tab2 , tab3 ,tab4= st.tabs(["📤 Upload & View", "🔍 Detection & Result","MULTI","random"])
        
with tab1:
    # File uploader
    uploaded_file = st.file_uploader("Drag and drop file here", type=["dcm"], label_visibility="collapsed")
    
    if uploaded_file:
        try:
            #preparing
            dcm= pydicom.dcmread(uploaded_file)
            img= dcm.pixel_array.astype(float)
            img= rescale_image(img, dcm.RescaleSlope, dcm.RescaleIntercept)
            
            img= apply_window_policy(img)
            img -= img.min((0,1))
            final_image = (img*255).astype(np.uint8)
            
            st.session_state["final_image"] = final_image
            st.session_state["uploaded_file"] = uploaded_file
            
            col1,col2 = st.columns(2)
            
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
                    "Image Size": f"{img.shape}"
                    
                })
            
        
    
        except Exception as e:
            st.error(f"Error reading DICOM file: {e}")
    
    
with tab2:
    st.markdown("### 🧠 ICH Detection Simulation")

    from model import load_model, predict_slice
    
    if "final_image" in st.session_state:
        img = st.session_state.final_image
        
        mean_img = [0.22363983, 0.18190407, 0.2523437 ]
        std_img = [0.32451536, 0.2956294,  0.31335256]

        ''''inference_transform = A.Compose([
            A.Resize(480, 480),
            A.Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
            ToTensorV2()
            ])
'''
        if st.button("Run Detection"):
            #model = load_model()
#
            #score = predict_slice(model, img)
            #gradcam_img = generate_gradcam(img)
            #bbox_img, subtype, subtype_score = detect_hemorrhage(img)

            #st.image(gradcam_img, caption=f"🔍 Grad-CAM - Confidence: {score:.2f}", use_column_width=True)
            #st.image(bbox_img, caption=f"🩸 Subtype: {subtype} (Score: {subtype_score})", use_column_width=True)

            #if score > 0.6:
                st.success("✅ ICH detected in this slice.")
            #else:
                st.info("ℹ️ No ICH detected.")
    else:
        st.warning("Upload and process a DICOM slice in Tab 1 first.")
    


with tab3:
    uploaded_files = st.file_uploader("Upload multiple DICOM files", type=["dcm"], accept_multiple_files=True)

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
                    "position": getattr(dcm, "ImagePositionPatient", [0])[2] if hasattr(dcm, "ImagePositionPatient") else dcm.get("InstanceNumber", 0)
                })

            except Exception as e:
                st.warning(f"Skipping a file due to error: {e}")

        # Sort by position
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

from PIL import Image

with tab4:
    uploaded_files = st.file_uploader("Upload multiple JPEG slice images", type=["jpg"], accept_multiple_files=True)
    
    if uploaded_files:
        # ✅ Sort files by filename (e.g. slice_01.jpg, slice_02.jpg, ...)
        uploaded_files = sorted(uploaded_files, key=lambda x: x.name)

        slices = []
        for file in uploaded_files:
            image = Image.open(file)
            slices.append({
                "filename": file.name,
                "image": image
            })

        # ✅ Slider navigation
        if slices:
            idx = st.slider("Navigate slices", 0, len(slices) - 1, 0)
            selected = slices[idx]
            st.image(selected["image"], caption=f"Slice {idx + 1}: {selected['filename']}", use_container_width=True)
