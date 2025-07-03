import os
import json
import logging
from typing import Dict, Any
from google import genai
from google.genai import types


class MedicalChatbot:
    def __init__(self):
        """Initialize the Medical Chatbot with Gemini API"""
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"
        
        # Medical departments mapping
        self.medical_departments = {
            "cardiology": "🫀 **Cardiology** - Specializes in heart and cardiovascular system disorders",
            "neurology": "🧠 **Neurology** - Focuses on brain, spinal cord, and nervous system conditions",
            "orthopedics": "🦴 **Orthopedics** - Treats bones, joints, ligaments, and musculoskeletal system",
            "dermatology": "🌟 **Dermatology** - Handles skin, hair, and nail conditions",
            "ophthalmology": "👁️ **Ophthalmology** - Specializes in eye and vision problems",
            "ent": "👂 **ENT (Otolaryngology)** - Treats ear, nose, throat, and related structures",
            "pulmonology": "🫁 **Pulmonology** - Focuses on respiratory system and lung diseases",
            "internal_medicine": "🩺 **Internal Medicine** - General adult medicine and internal organ systems",
            "gynecology": "🤰 **Gynecology** - Women's reproductive health and related conditions",
            "pediatrics": "👶 **Pediatrics** - Medical care for infants, children, and adolescents",
            "psychiatry": "🧘 **Psychiatry** - Mental health and psychological disorders",
            "gastroenterology": "🍽️ **Gastroenterology** - Digestive system and gastrointestinal disorders",
            "urology": "🩺 **Urology** - Urinary tract and male reproductive system",
            "endocrinology": "⚖️ **Endocrinology** - Hormonal disorders and endocrine system",
            "rheumatology": "🦴 **Rheumatology** - Autoimmune and inflammatory joint/muscle conditions",
            "emergency": "🚨 **Emergency Medicine** - Immediate medical attention required",
            "dentistry": "🦷 **Dentistry** - Dental and oral health issues"
        }
    
    def analyze_symptoms(self, user_input: str) -> str:
        """Analyze user symptoms and recommend appropriate medical department"""
        try:
            system_prompt = f"""
            You are a medical symptom analysis assistant. Your role is to:
            1. Analyze the user's symptoms or medical concerns
            2. Recommend the most appropriate medical department(s) to consult
            3. Provide clear explanations for your recommendations
            4. Offer general guidance on next steps
            
            Available medical departments:
            {json.dumps(list(self.medical_departments.keys()), indent=2)}
            
            Guidelines:
            - Be empathetic and supportive
            - Always recommend seeking professional medical advice
            - If symptoms suggest emergency, clearly state to seek immediate medical attention
            - Provide 1-2 most relevant department recommendations
            - Explain why each department is recommended
            - Include general advice about preparing for the medical visit
            - Never provide specific medical diagnoses or treatment advice
            
            Format your response as:
            1. Brief acknowledgment of their concern
            2. Recommended department(s) with explanations
            3. General next steps and preparation advice
            4. Reminder about professional medical consultation
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user", 
                        parts=[types.Part(text=f"System: {system_prompt}\n\nUser symptoms/concern: {user_input}")]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini API")
            
            # Enhance the response with department information
            enhanced_response = self._enhance_response_with_department_info(response.text)
            
            return enhanced_response
            
        except Exception as e:
            logging.error(f"Error in symptom analysis: {str(e)}")
            raise Exception(f"Failed to analyze symptoms: {str(e)}")
    
    def _enhance_response_with_department_info(self, response: str) -> str:
        """Enhance the response with detailed department information"""
        enhanced_response = response
        
        # Add department details if mentioned
        for dept_key, dept_info in self.medical_departments.items():
            if dept_key.lower() in response.lower():
                enhanced_response = enhanced_response.replace(
                    dept_key.capitalize(), 
                    dept_info
                )
        
        # Add footer with important reminders
        footer = """
        
---
        
**📝 Preparing for Your Medical Visit:**
- List all symptoms with timeline
- Note any medications you're taking
- Prepare questions for the doctor
- Bring relevant medical history
        
**⚠️ When to Seek Emergency Care:**
- Severe chest pain or difficulty breathing
- Signs of stroke (face drooping, arm weakness, speech difficulty)
- Severe allergic reactions
- Uncontrolled bleeding
- Loss of consciousness
        
**Remember:** This guidance is for informational purposes only. Always consult with healthcare professionals for proper medical evaluation and treatment.
        """
        
        return enhanced_response + footer
    
    def get_department_info(self, department: str) -> str:
        """Get detailed information about a specific medical department"""
        dept_key = department.lower().replace(" ", "_")
        return self.medical_departments.get(dept_key, "Department information not available.")
    
