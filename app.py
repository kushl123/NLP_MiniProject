import streamlit as st
import joblib
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

@st.cache_resource
def load_model_and_metadata():
    try:
        # Load the trained model and labels saved in Step 1
        model = joblib.load('toxic_classifier_svc.joblib')
        labels = joblib.load('labels.joblib')
        return model, labels
    except FileNotFoundError:
        st.error("Model files not found. Please run 'save_model.py' first.")
        return None, None


# Load the core components
SVC_pipeline, labels = load_model_and_metadata()

# Initialize the stemmer
stemmer = SnowballStemmer('english')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    # ... (include the rest of your cleaning regexes here)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def stemming_only(sentence):
    """Applies stemming to a clean string's tokens."""
    # We rely on the TfidfVectorizer's tokenization logic for the rest.
    stemmed_sentence = " ".join([stemmer.stem(word)
                                for word in sentence.split()])
    return stemmed_sentence

def analyze_toxicity(raw_comment, pipeline, labels):
    # 1. Apply cleaning and stemming to the raw input string
    cleaned_comment = clean_text(raw_comment)
    stemmed_comment = stemming_only(cleaned_comment)

    # 2. Pipeline transforms and predicts
    # Use decision_function for LinearSVC
    raw_scores = pipeline.decision_function([stemmed_comment])[0]

    # Use a safe threshold of 0 for SVC (which is the model's inherent decision boundary)
    predictions = (raw_scores > 0).astype(int)

    # 3. Compile results
    results = {label: score for label, score in zip(labels, raw_scores)}
    flagged_labels = [label for label, pred in zip(
        labels, predictions) if pred == 1]

    return flagged_labels, results, raw_scores

# --- Streamlit UI ---


st.set_page_config(page_title="Toxic Comment Classifier 💬", layout="centered")
st.title("💬 Toxic Comment Classifier")
st.markdown("Enter a comment below to check for toxicity across six categories.")

# Text Input Box
user_input = st.text_area("Comment to Analyze:", height=150)

if st.button("Analyze Comment", use_container_width=True):
    if SVC_pipeline is None:
        st.warning("Model failed to load. Check console for file errors.")
    elif not user_input.strip():
        st.info("Please enter a comment to analyze.")
    else:
        # Run the analysis
        flagged_labels, raw_results, raw_scores = analyze_toxicity(
            user_input, SVC_pipeline, labels)

        # --- Display Results ---
        score_df = pd.DataFrame(raw_results.items(), columns=[
            'Label', 'Raw Score'])

        # Notification on top right using custom HTML/CSS
        notification_style = """
            <style>
            .notif-box {
                position: fixed;
                top: 30px;
                right: 30px;
                z-index: 9999;
                padding: 18px 32px;
                border-radius: 8px;
                font-size: 1.2em;
                font-weight: bold;
                color: white;
                box-shadow: 0 2px 12px rgba(0,0,0,0.15);
                animation: fadein 0.5s;
            }
            .notif-red { background: #d32f2f; }
            .notif-green { background: #388e3c; }
            @keyframes fadein { from { opacity: 0; } to { opacity: 1; } }
            </style>
        """

        if flagged_labels:
            # # TOXIC OUTPUT
            # st.markdown(notification_style + f"""
            #     🚨 Toxic comment detected, please be respectful.
            # """, unsafe_allow_html=True)
            st.error("🚨 TOXIC COMMENT DETECTED 🚨")
            st.subheader(f"Flags Triggered: {', '.join(flagged_labels)}")

            # Strikethrough the input text to flag it visually
            st.markdown(f"**Input:** <span style='color:red'><s>{user_input}</s></span>", unsafe_allow_html=True)

            # Bar chart in red using Altair
            import altair as alt
            st.markdown("---")
            st.subheader("Toxicity Label Scores (Bar Chart)")
            chart = alt.Chart(score_df).mark_bar(color='#d32f2f').encode(
                x=alt.X('Label', sort=None),
                y='Raw Score',
                tooltip=['Label', 'Raw Score']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

            # Reasoning (Feature Importance): We use the raw decision scores as reasoning
            st.markdown("---")
            st.subheader("Model Decision Scores (Reasoning)")
            st.markdown(
                "Scores above 0.0 contribute to a 'Toxic' classification.")

            # Highlight scores that crossed the 0 threshold
            st.dataframe(score_df.style.apply(lambda x: ['background-color: #ffcccc' if x['Raw Score'] > 0 else '' for i in x], axis=1),
                         hide_index=True, use_container_width=True)

        else:
            # NON-TOXIC OUTPUT
            # st.markdown(notification_style + f"""
            #     <div class='notif-box notif-green'>✅ Comment not toxic.</div>
            # """, unsafe_allow_html=True)
            st.success("✅ Comment Classified as **Not Toxic**.")
            st.subheader("All categories below 0.0 threshold.")

            # Show input as normal (no strikethrough)
            st.markdown(f"**Input:** {user_input}")

            # Bar chart in green using Altair
            import altair as alt
            st.markdown("---")
            st.subheader("Toxicity Label Scores (Bar Chart)")
            chart = alt.Chart(score_df).mark_bar(color='#388e3c').encode(
                x=alt.X('Label', sort=None),
                y='Raw Score',
                tooltip=['Label', 'Raw Score']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

            # Display scores for non-toxic case
            st.dataframe(score_df, hide_index=True, use_container_width=True)
