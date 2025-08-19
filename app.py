import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import numpy as np
import joblib

# Cache the model and tokenizer to avoid reloading them on each run
@st.cache_resource
def load_model():
    model_name = "shiprocket-ai/open-llama-1b-address-completion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

@st.cache_resource
def load_classifier():
    model = joblib.load('address_classifier_model.pkl')
    vectorizer = joblib.load('address_vectorizer.pkl')
    return model, vectorizer

tokenizer, model = load_model()

def extract_address_components(address, max_new_tokens=150):
    """Extract address components using the model"""
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Extract address components from: {address}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05
        )
    
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()

def parse_extracted_address(extracted_str):
    try:
        start = extracted_str.find('{')
        end = extracted_str.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = extracted_str[start:end]
            return json.loads(json_str)
    except (json.JSONDecodeError, SyntaxError):
        return {}
    return {}

# --- UI Enhancements ---
st.set_page_config(layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: #000033;
    }
    
    .main-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #e6e6e6;
    }
    
    h1 {
        color: white;
        text-align: center;
        font-weight: 600;
    }
    
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1565C0;
    }
    
    .result-box {
        border: 1px solid #ddd;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Address Structuring and Completion Detection")

with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        address_input = st.text_area("Enter an address to check completion and its structure:", height=200,
                                     placeholder="e.g., 1600 Amphitheatre Parkway, Mountain View, CA 94043")
        
        if st.button("Analyze Address", use_container_width=True):
            if address_input:
                with st.spinner("AI is at work... 🤖"):
                    extracted_str = extract_address_components(address_input)
                    parsed_components = parse_extracted_address(extracted_str)
                    
                    st.session_state.parsed_components = parsed_components
                    st.session_state.address_input = address_input
            else:
                st.warning("Please enter an address to analyze.")
    
    with col2:
        st.image("https://images.pexels.com/photos/164558/pexels-photo-164558.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2", use_container_width=True)
        st.markdown("<p style='text-align: center;'>Your intelligent address assistant</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if 'parsed_components' in st.session_state:
    with st.container():
        st.markdown('<div class="main-container result-box">', unsafe_allow_html=True)
        
        if st.session_state.parsed_components:
            st.subheader("📝 Structured Address")
            st.json(st.session_state.parsed_components)
            
            st.subheader("✅ Address Completeness")
            classifier, vectorizer = load_classifier()
            vectorized = vectorizer.transform([st.session_state.address_input])
            prediction = classifier.predict(vectorized)[0]
            
            if prediction == 0:
                st.success("This address appears to be complete.")
            else:
                st.warning("This address may be incomplete.")
        else:
            st.error("Could not parse the address components. Please try a different address.")
            
        st.markdown('</div>', unsafe_allow_html=True)
