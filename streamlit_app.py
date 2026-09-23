import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st

from src.model import MultiClassSiameseUNet


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GeoVision AI Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# ORIGINAL GEOVISION COLORS
# ============================================================

COLOR_MAP = {
    0: [0, 0, 0],          # No Change
    1: [239, 68, 68],      # Building - Red
    2: [34, 197, 94],      # Vegetation - Green
    3: [234, 179, 8],      # Road - Yellow
    4: [59, 130, 246]      # Water - Blue
}

CLASS_NAMES = {
    0: "No Change",
    1: "Building Change",
    2: "Vegetation Change",
    3: "Road Infrastructure",
    4: "Water Body Expansion"
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans',
    -apple-system,
    BlinkMacSystemFont,
    sans-serif;
}

.stApp {
    background-color: #f8fafc;
}

/* Remove excessive Streamlit padding */
.block-container {
    padding-top: 1.2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* =========================================================
   HEADER
   ========================================================= */

.main-title-container {
    background:
        linear-gradient(
            135deg,
            #f97316 0%,
            #ea580c 50%,
            #c2410c 100%
        );

    padding: 22px 32px;
    border-radius: 16px;

    box-shadow:
        0 10px 25px -5px
        rgba(234, 88, 12, 0.35);

    margin-bottom: 22px;

    color: white;
}

.main-title-container h1 {
    font-size: 30px;
    font-weight: 800;
    color: white;
    margin: 0;
}

/* =========================================================
   NAVIGATION
   ========================================================= */

div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

.nav-button button {
    border-radius: 12px;
    font-weight: 800;
    color: #ea580c;
    border: 1.5px solid rgba(234, 88, 12, 0.25);
    background: white;
}

.nav-button button:hover {
    background: rgba(234, 88, 12, 0.10);
    color: #c2410c;
}

/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 18px;
    font-weight: 800;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 8px;
    margin-bottom: 15px;
}

/* =========================================================
   CARDS
   ========================================================= */

.home-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 6px solid #ea580c;
    border-radius: 16px;
    padding: 22px;

    box-shadow:
        0 8px 20px rgba(0, 0, 0, 0.04);
}

.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 6px solid #ea580c;
    border-radius: 16px;
    padding: 20px;

    box-shadow:
        0 8px 20px rgba(0, 0, 0, 0.04);
}

/* =========================================================
   UPLOAD AREA
   ========================================================= */

[data-testid="stFileUploader"] {
    border: 2px dashed #fb923c;
    border-radius: 16px;
    background: #ffffff;
    padding: 8px;

    box-shadow:
        0 4px 16px
        rgba(234, 88, 12, 0.08);
}

[data-testid="stFileUploader"]:hover {
    border-color: #ea580c;
}

/* =========================================================
   ORANGE BUTTON
   ========================================================= */

.stButton > button {
    background:
        linear-gradient(
            135deg,
            #f97316 0%,
            #ea580c 100%
        );

    color: white;

    font-weight: 700;
    font-size: 16px;

    border: none;
    border-radius: 10px;

    padding: 12px 25px;

    box-shadow:
        0 6px 18px
        rgba(234, 88, 12, 0.30);

    transition: 0.2s;
}

.stButton > button:hover {
    background:
        linear-gradient(
            135deg,
            #ea580c 0%,
            #c2410c 100%
        );

    color: white;

    transform: translateY(-1px);

    box-shadow:
        0 8px 24px
        rgba(234, 88, 12, 0.40);
}

/* =========================================================
   METRICS
   ========================================================= */

.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 6px solid #ea580c;
    border-radius: 14px;
    padding: 18px;
    text-align: center;

    box-shadow:
        0 5px 15px
        rgba(0, 0, 0, 0.04);
}

.metric-number {
    color: #ea580c;
    font-size: 28px;
    font-weight: 800;
}

.metric-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
}

/* =========================================================
   STATUS
   ========================================================= */

.status-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #ea580c;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 15px;
}

/* =========================================================
   IMAGE
   ========================================================= */

.result-image {
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding-top: 30px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "results" not in st.session_state:
    st.session_state.results = None

if "before_image" not in st.session_state:
    st.session_state.before_image = None

if "after_image" not in st.session_state:
    st.session_state.after_image = None

if "status" not in st.session_state:
    st.session_state.status = "Ready for analysis."


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = MultiClassSiameseUNet(
        num_classes=5
    ).to(device)

    if os.path.exists(
        "multiclass_siamese_model.pth"
    ):

        model_path = (
            "multiclass_siamese_model.pth"
        )

    elif os.path.exists(
        "best_siamese_model.pth"
    ):

        model_path = (
            "best_siamese_model.pth"
        )

    else:

        st.error(
            "Model file not found."
        )

        st.stop()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    model.load_state_dict(
        state_dict,
        strict=False
    )

    del state_dict

    model.eval()

    return model, device


model, device = load_model()


# ============================================================
# READ IMAGE
# ============================================================

def prepare_image(uploaded_file):

    if uploaded_file is None:
        return None

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ============================================================
# IMAGE DIFFERENCE
# ============================================================

def calculate_cva_difference(
    before,
    after
):

    before_float = (
        before.astype(np.float32)
    )

    after_float = (
        after.astype(np.float32)
    )

    difference = np.linalg.norm(
        before_float - after_float,
        axis=2
    )

    max_difference = (
        np.max(difference) + 1e-8
    )

    normalized = (
        difference /
        max_difference
    )

    # Smooth small pixel noise
    normalized = cv2.GaussianBlur(
        normalized,
        (5, 5),
        0
    )

    return normalized


# ============================================================
# REMOVE SMALL NOISE
# ============================================================

def clean_change_mask(
    binary_mask
):

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        mask
    )

    MIN_AREA = 12

    for label in range(
        1,
        number_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_AREA:

            cleaned[
                labels == label
            ] = 255

    return cleaned


# ============================================================
# MAIN CHANGE DETECTION PIPELINE
# ============================================================

def run_full_pipeline(
    img_before,
    img_after
):

    if (
        img_before is None
        or img_after is None
    ):

        return None

    # --------------------------------------------------------
    # STEP 1
    # Resize exactly as the original application
    # --------------------------------------------------------

    before = cv2.resize(
        img_before,
        (256, 256)
    )

    after = cv2.resize(
        img_after,
        (256, 256)
    )

    # --------------------------------------------------------
    # STEP 2
    # Prepare tensors
    # --------------------------------------------------------

    before_tensor = torch.from_numpy(
        before.astype(
            np.float32
        ) / 255.0
    ).permute(
        2,
        0,
        1
    ).unsqueeze(
        0
    ).to(device)

    after_tensor = torch.from_numpy(
        after.astype(
            np.float32
        ) / 255.0
    ).permute(
        2,
        0,
        1
    ).unsqueeze(
        0
    ).to(device)

    # --------------------------------------------------------
    # STEP 3
    # Neural network prediction
    # --------------------------------------------------------

    with torch.inference_mode():

        output = model(
            before_tensor,
            after_tensor
        )

        probs = torch.softmax(
            output,
            dim=1
        ).squeeze(
            0
        ).cpu().numpy()

        model_prediction = np.argmax(
            probs,
            axis=0
        ).astype(
            np.int32
        )

    # --------------------------------------------------------
    # STEP 4
    # Actual Before/After difference
    # --------------------------------------------------------

    difference = calculate_cva_difference(
        before,
        after
    )

    # --------------------------------------------------------
    # STEP 5
    # Adaptive threshold
    # --------------------------------------------------------

    difference_uint8 = (
        difference * 255
    ).astype(
        np.uint8
    )

    _, cva_mask = cv2.threshold(
        difference_uint8,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # STEP 6
    # Clean noise
    # --------------------------------------------------------

    cva_mask = clean_change_mask(
        cva_mask
    )

    actual_change = (
        cva_mask > 0
    )

    # --------------------------------------------------------
    # STEP 7
    # Model confidence
    # --------------------------------------------------------

    model_changed = (
        model_prediction > 0
    )

    if probs.shape[0] > 1:

        change_probability = np.max(
            probs[1:],
            axis=0
        )

    else:

        change_probability = np.zeros(
            (256, 256),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # STEP 8
    # Start with the model result
    # --------------------------------------------------------

    predictions = (
        model_prediction.copy()
    )

    # --------------------------------------------------------
    # STEP 9
    # Model confidently detected change
    # --------------------------------------------------------

    confident_change = (
        model_changed
        & (change_probability >= 0.40)
    )

    # --------------------------------------------------------
    # STEP 10
    # Detect changes missed by model
    # --------------------------------------------------------

    missed_change = (
        actual_change
        & ~confident_change
    )

    # Most probable semantic class
    # among change classes

    if probs.shape[0] > 1:

        semantic_prediction = (
            np.argmax(
                probs[1:],
                axis=0
            ) + 1
        )

        semantic_confidence = (
            np.max(
                probs[1:],
                axis=0
            )
        )

    else:

        semantic_prediction = np.ones(
            (256, 256),
            dtype=np.int32
        )

        semantic_confidence = np.zeros(
            (256, 256),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Assign semantic classes where supported
    # --------------------------------------------------------

    for class_id in range(
        1,
        5
    ):

        class_pixels = (
            missed_change
            & (
                semantic_prediction
                == class_id
            )
            & (
                semantic_confidence
                >= 0.25
            )
        )

        predictions[
            class_pixels
        ] = class_id

    # --------------------------------------------------------
    # Remaining physical changes
    # are marked as Other Change
    # --------------------------------------------------------

    remaining_change = (
        missed_change
        & (predictions == 0)
    )

    predictions[
        remaining_change
    ] = 4

    # --------------------------------------------------------
    # IMPORTANT:
    # Don't turn the entire image into change.
    # --------------------------------------------------------

    predictions[
        ~actual_change
        & ~model_changed
    ] = 0

    # --------------------------------------------------------
    # Semantic color map
    # --------------------------------------------------------

    class_rgb = np.zeros(
        (
            256,
            256,
            3
        ),
        dtype=np.uint8
    )

    for class_id, color in COLOR_MAP.items():

        class_rgb[
            predictions == class_id
        ] = color

    # --------------------------------------------------------
    # Binary change mask
    # --------------------------------------------------------

    final_change_mask = (
        predictions > 0
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # Feature attention heatmap
    # --------------------------------------------------------

    combined_heat = (
        0.55
        * change_probability
        +
        0.45
        * difference
    )

    combined_heat = np.clip(
        combined_heat,
        0,
        1
    )

    heat_uint8 = (
        combined_heat * 255
    ).astype(
        np.uint8
    )

    heat_uint8 = cv2.GaussianBlur(
        heat_uint8,
        (9, 9),
        0
    )

    attention = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET
    )

    attention = cv2.cvtColor(
        attention,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Baseline difference map
    # --------------------------------------------------------

    baseline_uint8 = (
        difference * 255
    ).astype(
        np.uint8
    )

    baseline = cv2.applyColorMap(
        baseline_uint8,
        cv2.COLORMAP_JET
    )

    baseline = cv2.cvtColor(
        baseline,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_pixels = (
        predictions.size
    )

    changed_pixels = int(
        np.sum(
            predictions > 0
        )
    )

    change_percentage = (
        changed_pixels
        / total_pixels
    ) * 100

    # Original project uses
    # 2.25 m² per pixel

    PIXEL_AREA = 2.25

    class_pixel_counts = {}
    class_percentages = {}
    surface_area = {}

    for class_id, class_name in (
        CLASS_NAMES.items()
    ):

        count = int(
            np.sum(
                predictions
                == class_id
            )
        )

        percentage = (
            count
            / total_pixels
        ) * 100

        class_pixel_counts[
            class_name
        ] = count

        class_percentages[
            class_name
        ] = percentage

        surface_area[
            class_name
        ] = (
            count
            * PIXEL_AREA
        )

    # --------------------------------------------------------
    # Total changed area
    # --------------------------------------------------------

    total_changed_area = (
        changed_pixels
        * PIXEL_AREA
    )

    return {

        "before": before,

        "after": after,

        "mask": final_change_mask,

        "class_map": class_rgb,

        "attention": attention,

        "baseline": baseline,

        "predictions": predictions,

        "cva_mask": cva_mask,

        "difference": difference,

        "change_percentage":
            change_percentage,

        "changed_pixels":
            changed_pixels,

        "total_changed_area":
            total_changed_area,

        "class_pixel_counts":
            class_pixel_counts,

        "class_percentages":
            class_percentages,

        "surface_area":
            surface_area
    }


# ============================================================
# PNG CONVERSION
# ============================================================

def image_to_png_bytes(
    image
):

    if image is None:
        return None

    if len(image.shape) == 3:

        bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    else:

        bgr = image

    success, encoded = cv2.imencode(
        ".png",
        bgr
    )

    if not success:
        return None

    return encoded.tobytes()


# ============================================================
# ANALYTICS CHART
# ============================================================

def generate_analytics_chart(
    results
):

    categories = [
        "Building",
        "Vegetation",
        "Road",
        "Water"
    ]

    areas = [

        results[
            "surface_area"
        ]["Building Change"],

        results[
            "surface_area"
        ]["Vegetation Change"],

        results[
            "surface_area"
        ]["Road Infrastructure"],

        results[
            "surface_area"
        ]["Water Body Expansion"]
    ]

    chart_colors = [
        "#ef4444",
        "#22c55e",
        "#eab308",
        "#3b82f6"
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        dpi=120
    )

    fig.patch.set_facecolor(
        "#0f172a"
    )

    # --------------------------------------------------------
    # Bar chart
    # --------------------------------------------------------

    axes[0].set_facecolor(
        "#0f172a"
    )

    axes[0].bar(
        categories,
        areas,
        color=chart_colors,
        edgecolor="white",
        linewidth=1
    )

    axes[0].set_title(
        "Ground Surface Change (m²)",
        color="white",
        fontsize=12,
        fontweight="bold"
    )

    axes[0].set_ylabel(
        "Area (m²)",
        color="#94a3b8"
    )

    axes[0].tick_params(
        colors="#94a3b8"
    )

    axes[0].grid(
        axis="y",
        linestyle="--",
        alpha=0.2
    )

    # --------------------------------------------------------
    # Pie chart
    # --------------------------------------------------------

    axes[1].set_facecolor(
        "#0f172a"
    )

    active = [
        (
            category,
            area,
            color
        )
        for category,
        area,
        color in zip(
            categories,
            areas,
            chart_colors
        )
        if area > 0
    ]

    if active:

        active_categories = [
            x[0]
            for x in active
        ]

        active_areas = [
            x[1]
            for x in active
        ]

        active_colors = [
            x[2]
            for x in active
        ]

        wedges, _, _ = axes[1].pie(
            active_areas,
            colors=active_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.75,
            textprops={
                "color": "white",
                "weight": "bold"
            }
        )

        axes[1].legend(
            wedges,
            active_categories,
            title="Change Type",
            loc="center left",
            bbox_to_anchor=(
                0.95,
                0.5
            ),
            frameon=False
        )

    else:

        axes[1].pie(
            [1],
            labels=[
                "No Change Detected"
            ],
            colors=[
                "#64748b"
            ]
        )

    axes[1].set_title(
        "Change Distribution Ratio",
        color="white",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title-container">
        <h1>
            Welcome to GeoVision AI Platform
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:

    if st.button(
        "Home",
        key="nav_home",
        use_container_width=True
    ):

        st.session_state.page = "Home"
        st.rerun()

with nav2:

    if st.button(
        "Upload Scenes",
        key="nav_upload",
        use_container_width=True
    ):

        st.session_state.page = (
            "Upload Scenes"
        )

        st.rerun()

with nav3:

    if st.button(
        "Detection Results",
        key="nav_results",
        use_container_width=True
    ):

        st.session_state.page = (
            "Detection Results"
        )

        st.rerun()

with nav4:

    if st.button(
        "Analytics & Downloads",
        key="nav_analytics",
        use_container_width=True
    ):

        st.session_state.page = (
            "Analytics & Downloads"
        )

        st.rerun()


st.markdown("---")


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        '<div class="section-title">'
        'Analytical Surface Summary'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        [2, 1]
    )

    with col1:

        st.markdown(
            '<div class="home-card">',
            unsafe_allow_html=True
        )

        if st.session_state.results:

            results = (
                st.session_state.results
            )

            fig = (
                generate_analytics_chart(
                    results
                )
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:

            st.info(
                "Ready to analyze satellite imagery."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="home-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Operational Log"
        )

        if st.session_state.results:

            results = (
                st.session_state.results
            )

            st.write(
                "**Latest Surface Analysis:**"
            )

            st.write(
                f'{results["total_changed_area"]:.1f} '
                f'm² total change detected'
            )

            st.write(
                f'{results["change_percentage"]:.2f}% '
                f'total change'
            )

        else:

            st.write(
                "Ready to analyze satellite imagery."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# UPLOAD SCENES
# ============================================================

elif st.session_state.page == "Upload Scenes":

    st.markdown(
        '<div class="section-title">'
        'Upload Scenes'
        '</div>',
        unsafe_allow_html=True
    )

    before_col, after_col = (
        st.columns(2)
    )

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    with before_col:

        st.markdown(
            "**Before Satellite Image**"
        )

        before_file = st.file_uploader(
            "Drop Before Image Here",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ],
            key="before_file"
        )

        if before_file is not None:

            before_image = (
                prepare_image(
                    before_file
                )
            )

            if before_image is not None:

                st.session_state.before_image = (
                    before_image
                )

                st.image(
                    before_image,
                    use_container_width=True
                )

            else:

                st.error(
                    "Could not read the Before image."
                )

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    with after_col:

        st.markdown(
            "**After Satellite Image**"
        )

        after_file = st.file_uploader(
            "Drop After Image Here",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ],
            key="after_file"
        )

        if after_file is not None:

            after_image = (
                prepare_image(
                    after_file
                )
            )

            if after_image is not None:

                st.session_state.after_image = (
                    after_image
                )

                st.image(
                    after_image,
                    use_container_width=True
                )

            else:

                st.error(
                    "Could not read the After image."
                )

    st.markdown("")

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "Analyze Changes",
        key="analyze_button",
        use_container_width=True
    ):

        before_image = (
            st.session_state.before_image
        )

        after_image = (
            st.session_state.after_image
        )

        if before_image is None:

            st.error(
                "Please upload the Before Satellite Image."
            )

        elif after_image is None:

            st.error(
                "Please upload the After Satellite Image."
            )

        else:

            with st.spinner(
                "Analyzing satellite images..."
            ):

                try:

                    results = (
                        run_full_pipeline(
                            before_image,
                            after_image
                        )
                    )

                    st.session_state.results = (
                        results
                    )

                    st.session_state.status = (
                        "Analysis Complete"
                    )

                    st.success(
                        "Analysis completed successfully."
                    )

                except Exception as error:

                    st.error(
                        f"Analysis failed: {error}"
                    )

    st.markdown(
        f"""
        <div class="status-box">
            <b>Status:</b>
            {st.session_state.status}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DETECTION RESULTS
# ============================================================

elif st.session_state.page == "Detection Results":

    results = (
        st.session_state.results
    )

    if results is None:

        st.warning(
            "No analysis has been performed yet."
        )

        if st.button(
            "Go to Upload Scenes"
        ):

            st.session_state.page = (
                "Upload Scenes"
            )

            st.rerun()

    else:

        st.markdown(
            '<div class="section-title">'
            'Detection Results'
            '</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        m1, m2, m3 = st.columns(3)

        with m1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {results["change_percentage"]:.2f}%
                    </div>
                    <div class="metric-label">
                        Surface Coverage Changed
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {results["changed_pixels"]:,}
                    </div>
                    <div class="metric-label">
                        Changed Pixels
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {results["total_changed_area"]:.1f}
                    </div>
                    <div class="metric-label">
                        Estimated Changed Area (m²)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("")

        # ----------------------------------------------------
        # THREE ORIGINAL RESULT VIEWS
        # ----------------------------------------------------

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.markdown(
                "**5-Class Semantic Change Map**"
            )

            st.image(
                results["class_map"],
                use_container_width=True
            )

        with col2:

            st.markdown(
                "**Feature Attention Heatmap**"
            )

            st.image(
                results["attention"],
                use_container_width=True
            )

        with col3:

            st.markdown(
                "**Baseline Difference Map**"
            )

            st.image(
                results["baseline"],
                use_container_width=True
            )

        st.markdown("---")

        # ----------------------------------------------------
        # ORIGINAL INPUTS
        # ----------------------------------------------------

        st.markdown(
            "### Input Satellite Images"
        )

        input1, input2 = (
            st.columns(2)
        )

        with input1:

            st.caption(
                "Before Satellite Image"
            )

            st.image(
                results["before"],
                use_container_width=True
            )

        with input2:

            st.caption(
                "After Satellite Image"
            )

            st.image(
                results["after"],
                use_container_width=True
            )

        st.markdown("---")

        # ----------------------------------------------------
        # COLOR LEGEND
        # ----------------------------------------------------

        st.markdown(
            "### Change Classification"
        )

        legend_cols = st.columns(5)

        for index, (
            class_id,
            class_name
        ) in enumerate(
            CLASS_NAMES.items()
        ):

            color = (
                COLOR_MAP[class_id]
            )

            rgb_color = (
                f"rgb("
                f"{color[0]},"
                f"{color[1]},"
                f"{color[2]}"
                f")"
            )

            with legend_cols[index]:

                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #e2e8f0;
                        border-radius:10px;
                        padding:10px;
                        text-align:center;
                        background:white;
                    ">

                    <span style="
                        display:inline-block;
                        width:18px;
                        height:18px;
                        background:{rgb_color};
                        border-radius:4px;
                        vertical-align:middle;
                        margin-right:5px;
                    "></span>

                    <span style="
                        font-size:12px;
                        font-weight:600;
                    ">
                        {class_name}
                    </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# ANALYTICS & DOWNLOADS
# ============================================================

elif (
    st.session_state.page
    == "Analytics & Downloads"
):

    results = (
        st.session_state.results
    )

    if results is None:

        st.warning(
            "Run an analysis first."
        )

    else:

        st.markdown(
            '<div class="section-title">'
            'Analytics & Downloads'
            '</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # ANALYTICS
        # ----------------------------------------------------

        chart_col, log_col = (
            st.columns([2, 1])
        )

        with chart_col:

            st.markdown(
                '<div class="home-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### Analytical Surface Summary"
            )

            fig = (
                generate_analytics_chart(
                    results
                )
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        with log_col:

            st.markdown(
                '<div class="home-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### Operational Log"
            )

            st.write(
                "**Latest Surface Analysis:**"
            )

            st.write(
                f'{results["total_changed_area"]:.1f} '
                f'm² '
                f'({results["change_percentage"]:.2f}% '
                f'total change detected)'
            )

            st.write(
                "Semantic classification completed."
            )

            st.write(
                "Before/After comparison completed."
            )

            st.write(
                "Change mask generated."
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ----------------------------------------------------
        # AREA METRICS
        # ----------------------------------------------------

        st.markdown(
            "### Calculated Surface Area Metrics"
        )

        area = (
            results["surface_area"]
        )

        a1, a2, a3, a4 = (
            st.columns(4)
        )

        with a1:

            st.metric(
                "Building",
                f'{area["Building Change"]:.2f} m²'
            )

        with a2:

            st.metric(
                "Vegetation",
                f'{area["Vegetation Change"]:.2f} m²'
            )

        with a3:

            st.metric(
                "Road",
                f'{area["Road Infrastructure"]:.2f} m²'
            )

        with a4:

            st.metric(
                "Water",
                f'{area["Water Body Expansion"]:.2f} m²'
            )

        st.markdown("---")

        # ----------------------------------------------------
        # DOWNLOADS
        # ----------------------------------------------------

        st.markdown(
            "### Download Visual Image Exports"
        )

        d1, d2, d3 = (
            st.columns(3)
        )

        with d1:

            semantic_bytes = (
                image_to_png_bytes(
                    results["class_map"]
                )
            )

            if semantic_bytes is not None:

                st.download_button(
                    "Download Semantic Change Map",
                    semantic_bytes,
                    "satellite_change_map.png",
                    "image/png",
                    use_container_width=True
                )

        with d2:

            attention_bytes = (
                image_to_png_bytes(
                    results["attention"]
                )
            )

            if attention_bytes is not None:

                st.download_button(
                    "Download Attention Heatmap",
                    attention_bytes,
                    "feature_attention_heatmap.png",
                    "image/png",
                    use_container_width=True
                )

        with d3:

            baseline_bytes = (
                image_to_png_bytes(
                    results["baseline"]
                )
            )

            if baseline_bytes is not None:

                st.download_button(
                    "Download Difference Map",
                    baseline_bytes,
                    "baseline_difference_map.png",
                    "image/png",
                    use_container_width=True
                )

        # ----------------------------------------------------
        # DOWNLOAD ORIGINAL CHANGE MASK
        # ----------------------------------------------------

        mask_bytes = (
            image_to_png_bytes(
                results["mask"]
            )
        )

        if mask_bytes is not None:

            st.download_button(
                "Download Binary Change Mask",
                mask_bytes,
                "change_mask.png",
                "image/png",
                use_container_width=True
            )

        # ----------------------------------------------------
        # ANALYTICS CHART DOWNLOAD
        # ----------------------------------------------------

        chart_fig = (
            generate_analytics_chart(
                results
            )
        )

        chart_path = (
            "geovision_analytics_chart.png"
        )

        chart_fig.savefig(
            chart_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=chart_fig.get_facecolor()
        )

        plt.close(chart_fig)

        if os.path.exists(
            chart_path
        ):

            with open(
                chart_path,
                "rb"
            ) as chart_file:

                chart_bytes = (
                    chart_file.read()
                )

            st.download_button(
                "Download Analytics Breakdown Chart",
                chart_bytes,
                "analytics_breakdown_chart.png",
                "image/png",
                use_container_width=True
            )

        # ----------------------------------------------------
        # CHANGE DISTRIBUTION
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            "### Change Distribution"
        )

        for class_id, class_name in (
            CLASS_NAMES.items()
        ):

            percentage = (
                results[
                    "class_percentages"
                ][class_name]
            )

            if class_id == 0:
                continue

            st.write(
                f"**{class_name}:** "
                f"{percentage:.2f}%"
            )

            st.progress(
                min(
                    int(
                        percentage
                    ),
                    100
                )
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        GeoVision AI Platform •
        Satellite Image Change Detection
    </div>
    """,
    unsafe_allow_html=True
)