"""Streamlit interface for running the complaint analysis pipeline interactively."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_complaint

st.set_page_config(page_title="RootCause AI", layout="wide")

st.title("RootCause AI")
st.caption("Confidence-aware complaint intelligence system")

complaint = st.text_area(
    "Enter Complaint Text",
    height=220,
    placeholder="Example: A debt collector keeps contacting me about a debt that is not mine and refuses to provide validation.",
)

if st.button("Analyze Complaint", use_container_width=True):
    if complaint.strip():
        with st.spinner("Running dynamic complaint analysis..."):
            result = predict_complaint(complaint)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Product", result["predicted_product"])
        c2.metric("Issue", result["predicted_issue"])
        c3.metric("Confidence", f"{result['overall_confidence']*100:.1f}%")
        c4.metric("Band", result["confidence_band"])

        if result["confidence_band"] == "Low":
            st.error("Low confidence: manual review recommended.")
            st.markdown("**System Decision:** Flag for manual review")
        elif result["confidence_band"] == "Medium":
            st.warning("Medium confidence: review suggested before acceptance.")
            st.markdown("**System Decision:** Review suggested before acceptance")
        else:
            st.success("High confidence prediction.")
            st.markdown("**System Decision:** Auto-classification accepted")

        st.markdown("### AI Insights")
        if result["llm_status"] != "success":
            st.info(f"LLM status: {result['llm_status']}. {result['llm_error']}")
        st.write(f"**Summary:** {result['summary']}")
        st.write(f"**Urgency:** {result['urgency']}")
        st.write(f"**Explanation:** {result['explanation']}")
        st.write(f"**Recommended Action:** {result['recommended_action']}")
        st.write(f"**Case Note:** {result['case_note']}")

        st.markdown("### Similar Historical Complaints")
        for i, case in enumerate(result["similar_cases"], start=1):
            with st.expander(f"Similar Case {i} - Similarity {case['similarity']}"):
                st.write(f"**Product:** {case['product']}")
                st.write(f"**Issue:** {case['issue']}")
                text_preview = case["complaint_text"]
                if len(text_preview) > 700:
                    text_preview = text_preview[:700] + "..."
                st.write(text_preview)

        with st.expander("Top-3 Model Predictions"):
            st.write("**Products**")
            for name, probability in result["top3_products"]:
                st.write(f"- {name}: {probability*100:.1f}%")

            st.write("**Issues**")
            for name, probability in result["top3_issues"]:
                st.write(f"- {name}: {probability*100:.1f}%")

        with st.expander("Confidence Breakdown"):
            st.write(f"Product confidence: {result['product_confidence']*100:.2f}%")
            st.write(f"Issue confidence: {result['issue_confidence']*100:.2f}%")
            st.write(f"Product margin: {result['product_margin']*100:.2f}%")
            st.write(f"Compatibility score: {result['compatibility_score']*100:.2f}%")
            st.write(f"Average retrieval similarity: {result['retrieval_similarity']*100:.2f}%")
            st.write(f"Raw product prediction: {result['raw_product_prediction']}")
            st.write(f"Raw issue prediction: {result['raw_issue_prediction']}")
            st.write(f"Product-aware issue prediction: {result['product_aware_issue_prediction']}")
    else:
        st.warning("Please enter complaint text first.")
