import streamlit as st
from openai import OpenAI

# Load API Key from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit UI
st.title("🎨 AI Coloring Page Generator")

# User input for prompt
user_prompt = st.text_area("Enter your prompt:", placeholder="Describe the image you want as an outline...", height=100)

# Generate button
if st.button("Generate"):
    if user_prompt.strip():
        with st.spinner("🎨 Generating your image... Please wait!"):
            try:
                # Generate image using OpenAI API
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=f"You are an artist that creates templates for people to fill colors in themselves. "
                           f"Create an image with the following information: {user_prompt}. "
                           f"Make sure to only make outlines and don't fill in any colors.",
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )

                # Extract image URL
                image_url = response.data[0].url

                # Display image
                st.image(image_url, caption="🖼️ Your AI-Generated Coloring Image", use_column_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a prompt before generating.")

