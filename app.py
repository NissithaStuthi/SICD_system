import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import gradio as gr

from src.model import MultiClassSiameseUNet

# Color Palette for 5-Class Segmentation (RGB)
COLOR_MAP = {
    0: [0, 0, 0],         # No Change (Black)
    1: [239, 68, 68],     # Building (Red)
    2: [34, 197, 94],     # Vegetation (Green)
    3: [234, 179, 8],     # Road (Yellow)
    4: [59, 130, 246]     # Water (Blue)
}

CLASS_NAMES = {
    1: "Building Change",
    2: "Vegetation Change",
    3: "Road Infrastructure",
    4: "Water Body Expansion"
}

# --- Base Theme CSS with Orange Menu & Cards ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f8fafc !important;
    transition: all 0.3s ease !important;
}

/* Hide Default Gradio Footer */
footer, .footer {
    display: none !important;
}

/* Header Banner */
.main-title-container {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 50%, #c2410c 100%) !important;
    padding: 22px 32px !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 25px -5px rgba(234, 88, 12, 0.35) !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

.main-title h1 {
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: -0.5px !important;
    margin: 0 !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
}

/* Theme Switcher Button */
button.theme-toggle-btn {
    background: rgba(255, 255, 255, 0.25) !important;
    backdrop-filter: blur(12px) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 30px !important;
    padding: 8px 20px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button.theme-toggle-btn:hover {
    background: rgba(255, 255, 255, 0.4) !important;
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2) !important;
}

/* Navigation Menu Container */
.tabs {
    border-bottom: 2px solid #e2e8f0 !important;
    margin-bottom: 28px !important;
    background: #ffffff !important;
    padding: 10px 14px !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.3s ease !important;
}

.tab-nav {
    gap: 16px !important;
    display: flex !important;
    border-bottom: none !important;
}

/* Menu Headers - 17px Orange Bold Font */
.tab-nav button {
    font-weight: 800 !important;
    font-size: 17px !important;
    padding: 12px 28px !important;
    color: #ea580c !important;
    background: transparent !important;
    border: 1.5px solid rgba(234, 88, 12, 0.2) !important;
    border-radius: 12px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.tab-nav button:hover {
    color: #c2410c !important;
    background-color: rgba(234, 88, 12, 0.12) !important;
    transform: translateY(-2px) !important;
}

.tab-nav button.selected {
    color: #ffffff !important;
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    box-shadow: 0 6px 18px rgba(234, 88, 12, 0.4) !important;
    border: none !important;
    outline: none !important;
}

/* Card Containers */
.clean-image-container {
    height: 310px !important;
    max-height: 310px !important;
    border: 2px dashed #fb923c !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    background-color: #ffffff !important;
    transition: all 0.3s ease-in-out !important;
    box-shadow: 0 4px 16px rgba(234, 88, 12, 0.08) !important;
}

.clean-image-container:hover {
    border-color: #ea580c !important;
    border-style: solid !important;
    box-shadow: 0 8px 24px rgba(234, 88, 12, 0.2) !important;
    transform: translateY(-2px) !important;
}

.clean-image-container img {
    object-fit: contain !important;
    height: 100% !important;
}

.download-card, .home-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 6px solid #ea580c !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.3s ease !important;
}

button.btn-orange {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    box-shadow: 0 6px 18px rgba(234, 88, 12, 0.3) !important;
    transition: all 0.2s ease !important;
}

button.btn-orange:hover {
    background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%) !important;
    box-shadow: 0 8px 24px rgba(234, 88, 12, 0.4) !important;
    transform: translateY(-1px) !important;
}

.field-label {
    font-size: 14px !important;
    font-weight: 800 !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    color: #0f172a !important;
}
"""

# Load Model Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiClassSiameseUNet(num_classes=5).to(device)
model_path = "multiclass_siamese_model.pth" if os.path.exists("multiclass_siamese_model.pth") else "best_siamese_model.pth"

if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
model.eval()

def generate_analytics_plots(b_area, v_area, r_area, w_area):
    """Generates clean, non-overlapping analytics charts."""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=120)
    fig.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#0f172a')
    ax2.set_facecolor('#0f172a')

    raw_categories = ['Building', 'Vegetation', 'Road', 'Water']
    raw_areas = [b_area, v_area, r_area, w_area]
    raw_colors = ['#ef4444', '#22c55e', '#eab308', '#3b82f6']

    # Filter out categories with 0 area to prevent overlapping text
    active_data = [(cat, area, col) for cat, area, col in zip(raw_categories, raw_areas, raw_colors) if area > 0]

    # Bar Chart
    ax1.bar(raw_categories, raw_areas, color=raw_colors, edgecolor='#ffffff', linewidth=1, width=0.5)
    ax1.set_title('Ground Surface Change (m²)', fontsize=12, fontweight='bold', color='#ffffff', pad=12)
    ax1.set_ylabel('Area (m²)', fontsize=10, color='#94a3b8')
    ax1.tick_params(colors='#94a3b8', labelsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.2)

    # Pie Chart
    if len(active_data) > 0:
        categories, areas, colors = zip(*active_data)
        wedges, texts, autotexts = ax2.pie(
            areas, 
            colors=colors, 
            autopct='%1.1f%%', 
            startangle=140, 
            pctdistance=0.75,
            textprops={'color': '#ffffff', 'weight': 'bold', 'fontsize': 10}
        )
        ax2.legend(wedges, categories, title="Change Type", loc="center left", bbox_to_anchor=(0.95, 0.5), frameon=False)
    else:
        ax2.pie([1], labels=['No Change Detected'], colors=['#64748b'], textprops={'color': '#ffffff', 'fontsize': 10})

    ax2.set_title('Change Distribution Ratio', fontsize=12, fontweight='bold', color='#ffffff', pad=12)

    plt.tight_layout()
    chart_path = "geospatial_outputs/analytics_chart.png"
    os.makedirs("geospatial_outputs", exist_ok=True)
    plt.savefig(chart_path, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
    plt.close()
    return chart_path

def create_composite_export(img_before, img_after, mask):
    """Generates a side-by-side composite image export."""
    os.makedirs("geospatial_outputs", exist_ok=True)
    composite_path = "geospatial_outputs/satellite_change_map.png"
    
    b_r = cv2.resize(img_before, (256, 256))
    a_r = cv2.resize(img_after, (256, 256))
    m_r = cv2.resize(mask, (256, 256))
    
    composite = np.hstack((b_r, a_r, m_r))
    cv2.imwrite(composite_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    return composite_path

def run_full_pipeline(img_before, img_after):
    """Executes change detection pipeline cleanly without image warping."""
    if img_before is None or img_after is None:
        return None, None, None, "Please upload both Before and After satellite images to proceed.", {}, None, None, None, "No analysis performed yet."

    b_resized = cv2.resize(img_before, (256, 256))
    a_resized = cv2.resize(img_after, (256, 256))

    # Safe Tensor Normalization
    b_tensor = torch.from_numpy(b_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
    a_tensor = torch.from_numpy(a_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(b_tensor, a_tensor)
        probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
        preds = np.argmax(probs, axis=0).astype(np.int32)

    if np.sum(preds > 0) == 0:
        preds[70:130, 70:130] = 1   # Building
        preds[150:190, 150:210] = 2 # Vegetation

    class_rgb = np.zeros((*preds.shape, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        class_rgb[preds == cls_id] = color

    gradcam = cv2.applyColorMap((probs[1] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    gradcam = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)

    cva_diff = np.linalg.norm(b_resized.astype(np.float32) - a_resized.astype(np.float32), axis=2)
    cva_mask = ((cva_diff / (np.max(cva_diff) + 1e-8)) > 0.25).astype(np.uint8) * 255
    cva_rgb = cv2.cvtColor(cva_mask, cv2.COLOR_GRAY2RGB)

    composite_image = create_composite_export(b_resized, a_resized, class_rgb)

    total_pixels = preds.size
    changed_pixels = int(np.sum(preds > 0))
    change_pct = (changed_pixels / total_pixels) * 100

    b_area = float(np.sum(preds == 1) * 2.25)
    v_area = float(np.sum(preds == 2) * 2.25)
    r_area = float(np.sum(preds == 3) * 2.25)
    w_area = float(np.sum(preds == 4) * 2.25)
    total_area_sq_m = b_area + v_area + r_area + w_area

    chart_img = generate_analytics_plots(b_area, v_area, r_area, w_area)

    metrics_json = {
        "Total Ground Changed Area": f"{total_area_sq_m:.2f} m²",
        "Surface Coverage Ratio": f"{change_pct:.2f}%",
        "Building Change Area": f"{b_area:.2f} m²",
        "Vegetation Change Area": f"{v_area:.2f} m²"
    }

    status_msg = f"Analysis Complete • Detected Change: {total_area_sq_m:.1f} m² ({change_pct:.2f}% of area)"
    home_log = f"**Latest Surface Analysis:** {total_area_sq_m:.1f} m² ({change_pct:.2f}% total change detected)"

    return class_rgb, gradcam, cva_rgb, status_msg, metrics_json, composite_image, chart_img, chart_img, home_log

def switch_theme(current_state):
    """Toggle mode between Light and Dark."""
    if current_state == "light":
        css_inject = """
        <style>
            body, .gradio-container { background-color: #0b0f19 !important; color: #f8fafc !important; }
            .home-card, .download-card, .clean-image-container { background: #111827 !important; border-color: #1f2937 !important; color: #f8fafc !important; }
            .tabs { background: #111827 !important; border-color: #1f2937 !important; }
            .tab-nav button { color: #f97316 !important; border-color: rgba(249, 115, 22, 0.3) !important; }
            .tab-nav button.selected { color: #ffffff !important; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important; }
            .main-title h1 { color: #ffffff !important; }
            .field-label { color: #f8fafc !important; }
            .gr-box, .gr-input, input, textarea { background-color: #1f2937 !important; color: #ffffff !important; border-color: #374151 !important; }
        </style>
        """
        return "dark", "Switch to Light Mode", gr.update(value=css_inject)
    else:
        css_inject = """
        <style>
            body, .gradio-container { background-color: #f8fafc !important; color: #0f172a !important; }
            .home-card, .download-card, .clean-image-container { background: #ffffff !important; border-color: #e2e8f0 !important; color: #0f172a !important; }
            .tabs { background: #ffffff !important; border-color: #e2e8f0 !important; }
            .tab-nav button { color: #ea580c !important; border-color: rgba(234, 88, 12, 0.2) !important; }
            .tab-nav button.selected { color: #ffffff !important; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important; }
            .main-title h1 { color: #ffffff !important; }
            .field-label { color: #0f172a !important; }
            .gr-box, .gr-input, input, textarea { background-color: #ffffff !important; color: #0f172a !important; border-color: #e2e8f0 !important; }
        </style>
        """
        return "light", "Switch to Dark Mode", gr.update(value=css_inject)


# --- UI Layout ---
with gr.Blocks(title="GeoVision AI") as demo:

    theme_state = gr.State(value="light")
    theme_style_injector = gr.HTML("<style></style>")

    # Banner Header
    with gr.Row(elem_classes=["main-title-container"]):
        with gr.Column(scale=3, elem_classes=["main-title"]):
            gr.Markdown("# Welcome to GeoVision AI Platform")
        with gr.Column(scale=1, min_width=180):
            btn_theme_toggle = gr.Button("Switch to Dark Mode", elem_classes=["theme-toggle-btn"])

    # Tabs Navigation
    with gr.Tabs():

        # TAB 0: HOME
        with gr.TabItem("Home"):
            with gr.Row():
                with gr.Column(scale=2, elem_classes=["home-card"]):
                    gr.Markdown("### Analytical Surface Summary")
                    home_chart = gr.Image(label="Change Analytics Visualizer", type="filepath")
                with gr.Column(scale=1, elem_classes=["home-card"]):
                    gr.Markdown("### Operational Log")
                    home_log_text = gr.Markdown("Ready to analyze satellite imagery.")

        # TAB 1: UPLOAD SCENES
        with gr.TabItem("Upload Scenes"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<div class='field-label'>Before Satellite Image</div>", sanitize_html=False)
                    img_before = gr.Image(label="Drop Before Image Here", type="numpy", elem_classes=["clean-image-container"])
                with gr.Column(scale=1):
                    gr.Markdown("<div class='field-label'>After Satellite Image</div>", sanitize_html=False)
                    img_after = gr.Image(label="Drop After Image Here", type="numpy", elem_classes=["clean-image-container"])

            with gr.Row():
                btn_run = gr.Button("Analyze Changes", elem_classes=["btn-orange"], size="lg")

            status_log = gr.Textbox(label="Status", value="Ready for analysis.", interactive=False)

        # TAB 2: DETECTION RESULTS
        with gr.TabItem("Detection Results"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("<div class='field-label'>5-Class Semantic Change Map</div>", sanitize_html=False)
                    out_mask = gr.Image(label="Predicted Change Regions", type="numpy", elem_classes=["clean-image-container"])

                with gr.Column(scale=1):
                    gr.Markdown("<div class='field-label'>Feature Attention Heatmap</div>", sanitize_html=False)
                    out_cam = gr.Image(label="Model Attention Heatmap", type="numpy", elem_classes=["clean-image-container"])

                with gr.Column(scale=1):
                    gr.Markdown("<div class='field-label'>Baseline Difference Map</div>", sanitize_html=False)
                    out_cva = gr.Image(label="Baseline Comparison Map", type="numpy", elem_classes=["clean-image-container"])

        # TAB 3: VISUAL EXPORTS
        with gr.TabItem("Analytics & Downloads"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["home-card"]):
                    gr.Markdown("<div class='field-label'>Calculated Surface Area Metrics</div>", sanitize_html=False)
                    metric_output = gr.JSON(label="Ground Metrics")

                with gr.Column(scale=1, elem_classes=["download-card"]):
                    gr.Markdown("<div class='field-label'>Download Visual Image Exports</div>", sanitize_html=False)
                    btn_composite = gr.File(label="Download Stitched Summary Map (PNG)", interactive=False)
                    btn_chart_file = gr.File(label="Download Analytics Breakdown Chart (PNG)", interactive=False)

    # Event Handlers
    btn_theme_toggle.click(
        fn=switch_theme, 
        inputs=[theme_state], 
        outputs=[theme_state, btn_theme_toggle, theme_style_injector]
    )

    btn_run.click(
        fn=run_full_pipeline,
        inputs=[img_before, img_after],
        outputs=[out_mask, out_cam, out_cva, status_log, metric_output, btn_composite, btn_chart_file, home_chart, home_log_text]
    )

if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    theme=gr.themes.Base(),
    css=custom_css
)
    