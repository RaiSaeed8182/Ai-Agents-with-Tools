"""
Streamlit App for News Topic Classifier
Deploy the fine-tuned BERT model for live predictions
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Page config
st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="wide"
)

# Model path
MODEL_PATH = "./news_classifier_model"

# Label names
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer"""
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found! Please run train_news_classifier.py first.")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_topic(headline, model, tokenizer):
    """Predict news topic for a headline"""
    # Tokenize
    inputs = tokenizer(
        headline,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, probabilities[0].tolist()

# Main UI
st.title("📰 News Topic Classifier")
st.markdown("**Classify news headlines into: World, Sports, Business, or Sci/Tech**")
st.markdown("---")

# Load model
model, tokenizer = load_model()

if model is None or tokenizer is None:
    st.stop()

# Input section
st.subheader("🔍 Enter News Headline")
headline = st.text_area(
    "Headline:",
    placeholder="Enter a news headline to classify...",
    height=100
)

if st.button("🎯 Classify", type="primary"):
    if headline.strip():
        with st.spinner("Classifying..."):
            predicted_class, probabilities = predict_topic(headline, model, tokenizer)
            predicted_label = LABEL_NAMES[predicted_class]
            confidence = probabilities[predicted_class]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Main prediction
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"**Predicted Topic: {predicted_label}**")
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        
        # Probability distribution
        st.markdown("**Probability Distribution:**")
        prob_dict = {LABEL_NAMES[i]: probabilities[i] for i in range(4)}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        for label, prob in sorted_probs:
            percentage = prob * 100
            st.progress(prob, text=f"{label}: {percentage:.2f}%")
        
        # Show headline
        st.markdown("---")
        st.markdown(f"**Headline:** *{headline}*")
    else:
        st.warning("⚠️ Please enter a headline first!")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a fine-tuned **BERT-base-uncased** model 
    to classify news headlines into 4 categories:
    
    - 🌍 **World**
    - ⚽ **Sports**
    - 💼 **Business**
    - 🔬 **Sci/Tech**
    
    **Model:** BERT-base-uncased (fine-tuned on AG News Dataset)
    """)
    
    st.markdown("---")
    st.subheader("📝 Example Headlines")
    examples = [
        "Stock market reaches new high amid economic recovery",
        "Scientists discover new planet in habitable zone",
        "Local team wins championship in overtime thriller",
        "International summit addresses climate change crisis"
    ]
    
    for example in examples:
        if st.button(f"📌 {example[:50]}...", key=example):
            st.session_state.headline_example = example
            st.rerun()

# Auto-fill example if selected
if "headline_example" in st.session_state:
    headline = st.session_state.headline_example
    del st.session_state.headline_example
    st.rerun()

