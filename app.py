import os
import tempfile

import streamlit as st

from scripts.predict_pipeline import predict_demand

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Dress Demand Prediction",
    page_icon="👗",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title("👗 Dress Demand Prediction System")

st.write(
    """
    Upload a dress image and enter product rate
    to predict expected sales quantity.
    """
)

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Dress Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# RATE INPUT
# ==========================================

rate = st.number_input(
    "Enter Product Rate",
    min_value=100.0,
    value=500.0,
    step=50.0
)

# ==========================================
# RATE VALIDATION
# ==========================================

valid_rate = (
    100 <= rate <= 1500
)

if not valid_rate:

    st.warning(
        "Rate must be between 100 and 1500"
    )

# ==========================================
# PREDICT BUTTON
# ==========================================

predict_clicked = st.button(
    "Predict Demand",
    disabled=not valid_rate
)
# ==========================================
# PREDICT LOGIC
# ==========================================

if predict_clicked:

    # --------------------------------------
    # CHECK IMAGE
    # --------------------------------------

    if uploaded_file is None:

        st.error("Please upload an image")

    else:

        # ----------------------------------
        # SAVE TEMP IMAGE
        # ----------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.read()
            )

            temp_image_path = tmp_file.name

        # ----------------------------------
        # DISPLAY IMAGE
        # ----------------------------------

        st.image(
            temp_image_path,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ----------------------------------
        # LOADING
        # ----------------------------------

        with st.spinner("Analyzing image..."):

            result = predict_demand(
                temp_image_path,
                rate
            )

        # ----------------------------------
        # INVALID IMAGE
        # ----------------------------------

        if not result["success"]:

            st.error(result["message"])

        # ----------------------------------
        # VALID RESULT
        # ----------------------------------

        else:

            st.success("Prediction completed")

            st.subheader("Prediction Result")

            st.write(
                f"**Detected Category:** "
                f"{result['label']}"
            )

            st.write(
                f"**Validation Confidence:** "
                f"{result['confidence']:.4f}"
            )

            st.write(
                f"**Predicted Quantity Sold:** "
                f"{result['prediction']}"
            )
            
            # ======================================
            # SHOW SIMILAR DRESSES
            # ======================================

            st.subheader("Top Similar Dresses")

            similar_items = result["similar_items"]

            for item in similar_items:

                st.image(
                    item["image_path"],
                    width=180
                )

                st.write(
                    f"**Dress Code:** {item['code']}"
                )

                st.write(
                    f"**Historical Sales:** {item['qty']}"
                )

                st.write(
                    f"**Similarity Score:** "
                    f"{item['similarity']:.4f}"
                )

                st.markdown("---")

        # ----------------------------------
        # CLEANUP
        # ----------------------------------

        if os.path.exists(temp_image_path):

            os.remove(temp_image_path)