import streamlit as st
import json
from gemini_service import MedicalChatbot

# Page configuration
st.set_page_config(
    page_title="Medical Symptom Analyzer",
    page_icon="🏥",
    layout="wide"
)

# Initialize the medical chatbot
@st.cache_resource
def initialize_chatbot():
    return MedicalChatbot()

def main():
    st.title("🏥 Medical Symptom Analyzer")
    st.markdown("### Get guidance on which medical department to consult based on your symptoms")
    
    # Medical disclaimer
    st.warning("""
    ⚠️ **Important Medical Disclaimer**: 
    This tool provides general guidance only and is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare professionals for medical concerns. In case of emergency, contact emergency services immediately.
    """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chatbot = initialize_chatbot()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe your symptoms or ask a medical question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your symptoms..."):
                try:
                    response = st.session_state.chatbot.analyze_symptoms(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"❌ **Error**: Unable to process your request. Please try again later.\n\nDetails: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Sidebar with additional information
    with st.sidebar:
        st.header("📋 Medical Departments")
        st.markdown("""
        **Common departments we can recommend:**
        - 🫀 **Cardiology** - Heart and cardiovascular issues
        - 🧠 **Neurology** - Brain and nervous system
        - 🦴 **Orthopedics** - Bones, joints, and muscles
        - 🌟 **Dermatology** - Skin conditions
        - 👁️ **Ophthalmology** - Eye problems
        - 👂 **ENT** - Ear, nose, and throat
        - 🫁 **Pulmonology** - Respiratory issues
        - 🩺 **Internal Medicine** - General internal conditions
        - 🤰 **Gynecology** - Women's health
        - 👶 **Pediatrics** - Children's health
        - 🧘 **Psychiatry** - Mental health
        - 🦷 **Dentistry** - Dental and oral health
        """)
        
        st.header("🔄 Actions")
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        st.header("ℹ️ How it works")
        st.markdown("""
        1. **Describe your symptoms** in detail
        2. **AI analyzes** your input using advanced language models
        3. **Get recommendations** for the most appropriate medical department
        4. **Follow the guidance** to consult the right specialist
        """)

if __name__ == "__main__":
    main()