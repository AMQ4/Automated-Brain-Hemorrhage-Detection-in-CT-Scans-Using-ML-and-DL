import streamlit as st
#set page config
st.set_page_config(
    page_title="ICH Detection Assistant",
    page_icon="🧠",
    layout="centered"
    )
# Center the image using columns
col1, col2, col3 = st.columns([1, 2, 1])  # You can adjust the ratio

# Use the middle column for centering
with col2:
    img = "brain_logo.png"  
    st.image(img, width=200 ,use_container_width=True)


st.markdown("""
    <h1 style='text-align: center; font-size: 3em;'>ICH Detection Assistant</h1>
    """, unsafe_allow_html=True)

st.markdown("""
            <p style='text-align: center; font-size: 1.25em; color: #bbb;'>We care about your health. This app was developed to assist doctors and researchers in early detection of intracerebral hemorrhage.</p>
            """,unsafe_allow_html=True)



if st.button("Start", use_container_width=True):
    st.session_state.started = True
    st.success("Great! Now click **Detection** in the sidebar.")