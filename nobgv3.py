import streamlit as st
import io
from PIL import Image, ImageOps
import numpy as np
from cv2 import getStructuringElement, GaussianBlur, morphologyEx, MORPH_ELLIPSE, BORDER_DEFAULT
from scipy.ndimage import binary_erosion
from enum import Enum
from typing import Union, Optional, Any, Tuple
from rembg import new_session
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
import os
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
import base64
import datetime

kernel = getStructuringElement(MORPH_ELLIPSE, (3, 3))

class ReturnType(Enum):
    BYTES = 0
    PILLOW = 1
    NDARRAY = 2

def alpha_matting_cutout(
    img: Image.Image,
    mask: Image.Image,
    foreground_threshold: int,
    background_threshold: int,
    erode_structure_size: int,
) -> Image.Image:
    img_array = np.asarray(img)
    mask_array = np.asarray(mask)

    is_foreground = mask_array > foreground_threshold
    is_background = mask_array < background_threshold

    structure = np.ones(
        (erode_structure_size, erode_structure_size), dtype=np.uint8
    ) if erode_structure_size > 0 else None

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask_array.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img_array / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(cutout)

def post_process(mask: np.ndarray) -> np.ndarray:
    mask = morphologyEx(mask, MORPH_ELLIPSE, kernel)
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    return np.where(mask < 127, 0, 255).astype(np.uint8)

def fix_image_orientation(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img)

def remove(
    data: Union[bytes, Image.Image, np.ndarray],
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    only_mask: bool = False,
    post_process_mask: bool = False,
    bgcolor: Optional[Tuple[int, int, int, int]] = None,
    force_return_bytes: bool = False,
    *args: Optional[Any],
    **kwargs: Optional[Any]
) -> Union[bytes, Image.Image, np.ndarray]:
    if isinstance(data, bytes) or force_return_bytes:
        return_type = ReturnType.BYTES
        img = Image.open(io.BytesIO(data))
    elif isinstance(data, Image.Image):
        return_type = ReturnType.PILLOW
        img = data
    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = Image.fromarray(data)
    else:
        raise ValueError("Unsupported input type.")

    img = fix_image_orientation(img)

    # Create the session here instead of passing it as an argument
    session = new_session("u2net", *args, **kwargs)

    masks = session.predict(img, *args, **kwargs)
    cutouts = []

    for mask in masks:
        if post_process_mask:
            mask = Image.fromarray(post_process(np.array(mask)))

        if only_mask:
            cutout = mask
        elif alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                )
            except ValueError:
                cutout = Image.composite(img, Image.new("RGBA", img.size, 0), mask)
        else:
            cutout = Image.composite(img, Image.new("RGBA", img.size, 0), mask)

        cutouts.append(cutout)

    cutout = cutouts[0] if cutouts else img

    if return_type == ReturnType.PILLOW:
        return cutout
    elif return_type == ReturnType.NDARRAY:
        return np.asarray(cutout)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)
    return bio.read()

# Function to convert image to base64
def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Load logo
logo = Image.open("ClearCut/assets/ClearCut_Logo.png")  # Path to your logo
testo = Image.open("ClearCut/assets/footer_nobg.png")
# --- Page Configuration ---
# --- Set Page Configuration ---
st.set_page_config(
    page_title="Clear Cut",  # 📝 Change this to your desired tab title
    page_icon=os.path.join(os.path.dirname(__file__), "images", "Orange_logo.ico"),                   # 🎨 You can use an emoji or a URL to an icon
    layout="wide"                     # Optional: 'centered' or 'wide'
)
# --- Get Current Time ---
now = datetime.datetime.now()
hour = now.hour

# --- Dynamic Greeting Based on Time ---
if 5 <= hour < 12:
    greeting = "🌅 Good Morning!"
elif 12 <= hour < 18:
    greeting = "🌞 Good Afternoon!"
elif 18 <= hour < 22:
    greeting = "🌇 Good Evening!"
else:
    greeting = "🌙 Good Night!"

# --- Animated Greeting as Title ---
st.markdown(
    f"""
    <style>
    @keyframes fadeIn {{
      0% {{ opacity: 0; transform: translateY(-40px); }}
      100% {{ opacity: 1; transform: translateY(0); }}
    }}

    .greeting {{
      font-size: 50px;
      font-weight: bold;
      text-align: center;
      color: orange;
      animation: fadeIn 1s ease-in-out;
      margin-top: 20px;
    }}
    </style>

    <div class="greeting">{greeting}</div>
    """,
    unsafe_allow_html=True
)

# Add CSS for sidebar and content
st.markdown(
    """
    <style>
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #FF6600;  /* Orange color */
        color: white;
        padding: 20px;
        text-align: center;
    }
    .sidebar img {
        width: 100px;  /* Adjust the logo size */
        margin-bottom: 10px;
    }
    
    /* Main Content Styling */
    .main-content {
        margin-left: 250px;  /* Offset for sidebar */
        padding: 20px;
    }
    /* Centering the footer image */
    .footer {
        margin-top: 50px;
        text-align: center;
        max-width: 100%;
        width: 300px;  /* You can adjust the width if needed */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .footer img {
        max-width: 100%;
        width: 300px;  /* You can adjust the width if needed */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Optional: Add styling for the file upload button */
    .stFileUploader {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Convert logo image to base64
logo_base64 = image_to_base64(logo)

# Sidebar content (Logo and File Uploader)
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{logo_base64}" alt="Logo" />', unsafe_allow_html=True
)
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
footer_base64 = image_to_base64(testo)
# Footer
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{footer_base64}" alt="Footer Image" />', unsafe_allow_html=True
)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Title and functionality of the app
st.title("✨ Remove Image Background ")
st.write("Upload an image and remove its background instantly!")

# If an image is uploaded, show original vs processed images
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("## Original vs Processed Image")

    # Process the image (output_image_bytes from earlier steps)
    output_image_bytes = remove(uploaded_file.getvalue())  # Using the new remove function
    output_image = Image.open(io.BytesIO(output_image_bytes))

    # Layout: Side-by-side display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="🖼️ Original Image", use_container_width=True)
    with col2:
        st.image(output_image, caption="🌟 Background Removed", use_container_width=True)

    # Save output to Downloads folder
    downloads_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
    original_name = os.path.splitext(uploaded_file.name)[0]
    output_path = os.path.join(downloads_folder, f"{original_name}_nobg.png")

    st.success(f"✅ Background removed and will be saved as: {output_path}")

    # Add download button
    st.download_button(
        label="💾 Download Processed Image",
        data=output_image_bytes,
        file_name=f"{original_name}_nobg.png",
        mime="image/png"
    )

st.markdown('</div>', unsafe_allow_html=True)
