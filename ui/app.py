import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from sfs import horn_sfs

st.set_page_config(page_title="Lunar DEM Generator", layout="wide")
st.title("🌙 High-Resolution Lunar DEM Generator (Shape from Shading)")
st.markdown("""
Upload a lunar image, adjust SFS parameters, and generate a high-resolution Digital Elevation Model (DEM) using advanced Shape-from-Shading techniques.
""")

uploaded_file = st.file_uploader("Upload a lunar image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.sidebar.header("SFS Parameters")
    n_iter = st.sidebar.slider("Iterations", 100, 2000, 500, 100)
    lambd = st.sidebar.slider("Regularization (λ)", 0.0, 1.0, 0.2, 0.01)
    learning_rate = st.sidebar.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%e")
    lx = st.sidebar.slider("Light X", -1.0, 1.0, -1.0, 0.05)
    ly = st.sidebar.slider("Light Y", -1.0, 1.0, -1.0, 0.05)
    lz = st.sidebar.slider("Light Z", 0.0, 1.0, 0.5, 0.05)
    st.sidebar.header("Reference DEM (optional)")
    ref_file = st.sidebar.file_uploader("Upload Reference DEM (.npy or .tif)", type=["npy", "tif", "tiff"])
    ref_dem = None
    if ref_file:
        if ref_file.name.endswith('.npy'):
            ref_dem = np.load(ref_file)
        else:
            import rasterio
            with rasterio.open(ref_file) as src:
                ref_dem = src.read(1)
    if st.button("Generate DEM"):
        with st.spinner("Running advanced SFS..."):
            dem = horn_sfs(image_np, np.array([lx, ly, lz]), n_iterations=n_iter, lambd=lambd, learning_rate=learning_rate)
        st.success("DEM generated!")
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0].imshow(image_np, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        im = ax[1].imshow(dem, cmap='viridis')
        ax[1].set_title('DEM (Relative Height)')
        ax[1].axis('off')
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        st.pyplot(fig)
        # Download DEM
        buf = io.BytesIO()
        np.save(buf, dem)
        st.download_button("Download DEM (NumPy .npy)", buf.getvalue(), file_name="dem.npy")
        if ref_dem is not None and ref_dem.shape == dem.shape:
            error_map = dem - ref_dem
            rmse = np.sqrt(np.mean(error_map**2))
            mae = np.mean(np.abs(error_map))
            st.subheader("Reference DEM Comparison")
            st.write(f"**RMSE:** {rmse:.4f}, **MAE:** {mae:.4f}")
            fig_err, ax_err = plt.subplots()
            im_err = ax_err.imshow(error_map, cmap='coolwarm')
            ax_err.set_title('DEM Error Map (Generated - Reference)')
            plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
            st.pyplot(fig_err)
        elif ref_dem is not None:
            st.warning("Reference DEM shape does not match generated DEM. Skipping error map.")
        # Download DEM as GeoTIFF
        import rasterio
        from rasterio.transform import from_origin
        if st.button("Download DEM as GeoTIFF"):
            tiff_buf = io.BytesIO()
            transform = from_origin(0, 0, 1, 1)  # Dummy geotransform
            with rasterio.open(
                tiff_buf, 'w', driver='GTiff',
                height=dem.shape[0], width=dem.shape[1],
                count=1, dtype=dem.dtype, transform=transform
            ) as dst:
                dst.write(dem, 1)
            st.download_button("Download DEM (GeoTIFF)", tiff_buf.getvalue(), file_name="dem.tif")
        # PDF Report
        from fpdf import FPDF
        import tempfile
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, "Lunar DEM Generation Report", ln=True, align='C')
            pdf.ln(10)
            pdf.cell(200, 10, f"SFS Parameters: Iter={n_iter}, λ={lambd}, LR={learning_rate}", ln=True)
            if ref_dem is not None and ref_dem.shape == dem.shape:
                pdf.cell(200, 10, f"RMSE: {rmse:.4f}, MAE: {mae:.4f}", ln=True)
            pdf.ln(10)
            pdf.cell(200, 10, "See attached figures for DEM and error map.", ln=True)
            # Save and download
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            st.download_button("Download PDF Report", pdf_bytes, file_name="dem_report.pdf")
        # 3D Visualization
        import plotly.graph_objects as go
        st.subheader('3D DEM Visualization')
        x = np.arange(dem.shape[1])
        y = np.arange(dem.shape[0])
        X, Y = np.meshgrid(x, y)
        fig3d = go.Figure(data=[go.Surface(z=dem, x=X, y=Y, colorscale='Viridis')])
        fig3d.update_layout(title='3D DEM Surface', autosize=True, margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig3d, use_container_width=True)
else:
    st.info("Please upload a lunar image to begin.") 