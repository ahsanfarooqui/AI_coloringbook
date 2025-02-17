import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Load API Key from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API"]

# Initialize OpenAI client and LLM
client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4")

def generate_ideas(user_prompt):
    """Generates creative ideas for the image."""
    response = llm.invoke([
        {"role": "system", "content": "You are an AI that generates creative drawing ideas for coloring pages."},
        {"role": "user", "content": f"Suggest unique variations or related themes for this coloring page idea: {user_prompt}"}
    ])
    return response.content.strip()

def refine_prompt(idea):
    """Refines the prompt for better image generation."""
    return (f"You are an artist that creates templates for people to fill colors in themselves. "
            f"Create an image with the following information: {idea}. "
            f"Make sure to only make outlines and don't fill in any colors. "
            f"Avoid shading, keeping it simple with black and white lines only. "
            f"Ensure ample space between lines for easy coloring. Attention to detail is not very necessary.")

def generate_image(refined_prompt):
    """Generates the final image using DALL-E and returns the image URL."""
    response = client.images.generate(
        model="dall-e-3",
        prompt=refined_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url

# Define tools
idea_tool = Tool(
    name="Generate Ideas",
    func=generate_ideas,
    description="Generates creative drawing ideas based on user input."
)

refinement_tool = Tool(
    name="Refine Prompt",
    func=refine_prompt,
    description="Refines a given drawing idea into a structured prompt for better image generation."
)

image_tool = Tool(
    name="Generate Image",
    func=generate_image,
    description="Generates an AI coloring page based on the refined prompt. Returns a URL."
)

# Initialize Agent
agent = initialize_agent(
    tools=[idea_tool, refinement_tool, image_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.title("🎨 AI Coloring Page Generator (Agentic AI)")

# User input for prompt
user_prompt = st.text_area("Enter your prompt:", placeholder="Describe the image you want as an outline...", height=100)

# Generate button
if st.button("Generate"):
    if user_prompt.strip():
        log_messages = []
        
        with st.spinner("🤖 Generating image... Please wait!"):
            try:
                result = agent.run(f"Generate a creative idea, refine the prompt, and generate an image for this user prompt. Output should be only an image url: {user_prompt}")
                image_url = result
                #log_messages.append(f"Generated Idea: {idea}")
                #log_messages.append(f"Refined Prompt: {refined_prompt}")
                log_messages.append("Image successfully generated.")
                #st.write(f"**Generated Idea:** {idea}")
                st.image(image_url, caption="🖼️ Your AI-Generated Coloring Image", use_column_width=True)
            except Exception as e:
                error_message = f"❌ Error: {str(e)}"
                log_messages.append(error_message)
                st.error(error_message)
        
        # Display agent's decision-making log
        with st.expander("📝 Agent Decision Flow & Logs"):
            for log in log_messages:
                st.write(log)
    else:
        st.warning("⚠️ Please enter a prompt before generating.")

#-----------------
# import streamlit as st
# from openai import OpenAI

# # Load API Key from Streamlit Secrets
# OPENAI_API_KEY = st.secrets["OPENAI_API"]

# # Initialize OpenAI client
# client = OpenAI(api_key=OPENAI_API_KEY)

# # Streamlit UI
# st.title("🎨 AI Coloring Page Generator")

# # User input for prompt
# user_prompt = st.text_area("Enter your prompt:", placeholder="Describe the image you want as an outline...", height=100)

# # Generate button
# if st.button("Generate"):
#     if user_prompt.strip():
#         with st.spinner("🎨 Generating your image... Please wait!"):
#             try:
#                 # Generate image using OpenAI API
#                 response = client.images.generate(
#                     model="dall-e-3",
#                     prompt=f"You are an artist that creates templates for people to fill colors in themselves. "
#                            f"Create an image with the following information: {user_prompt}. "
#                            f"Make sure to only make outlines and don't fill in any colors. Also don't fill any shading or anything. simple black and white lines only."
#                             f"Make sure to keep ample space between the lines for people to fill in the colors."
#                             f"Attention to detail is not very necessary.",
#                     size="1024x1024",
#                     quality="standard",
#                     n=1,
#                 )

#                 # Extract image URL
#                 image_url = response.data[0].url

#                 # Display image
#                 st.image(image_url, caption="🖼️ Your AI-Generated Coloring Image", use_column_width=True)

#             except Exception as e:
#                 st.error(f"❌ Error: {str(e)}")
#     else:
#         st.warning("⚠️ Please enter a prompt before generating.")

