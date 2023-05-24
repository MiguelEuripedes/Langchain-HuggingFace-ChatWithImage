import streamlit as st
import os
from apikey import apikey
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool
from tempfile import NamedTemporaryFile, TemporaryDirectory

os.environ['OPENAI_API_KEY'] = apikey

tools = [ImageCaptionTool(), ObjectDetectionTool()]

# Creating chat and its memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    #penai_api_key='',
    temperature=0, # The response is more exact
    model="gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5, # The agent will only run 5 iterations
    verbose=True,
    memory=conversational_memory,
    early_stoppy_method = 'generate'
)

#set title
st.title("Ask a question about a image")

# set header
st.header("Upload a image")

# upload file
file = st.file_uploader("", type=["jpeg","jpg", "png"])

# Display image, if there is one
if file:
    st.image(file, use_column_width=True)

    # Question input
    user_question = st.text_input("Write your question:")

    # Save image to a temporary file
    with TemporaryDirectory() as temp_dir:
        with NamedTemporaryFile(dir=temp_dir, delete=True, mode='w+b') as f:
            f.write(file.getbuffer())
            image_path = f.name

            # Response
            if user_question and user_question != "":
                with st.spinner(text="Processing..."):
                    # Run agent response 
                    response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
            
                    st.write("Response:")
                    st.write(response)