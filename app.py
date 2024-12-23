import streamlit as st
import streamlit.components.v1 as components
from streamlit_feedback import streamlit_feedback
import datetime
import base64
from pathlib import Path
from log import logger
from pipeline import ModelPipeline
from session import session_handler, feedback_handler, history_handler
PROJECT_ROOT_DIR = str(Path(__file__).parent)


def get_session_history(session_id):
    history = history_handler.get_session_history(session_id)
    feedbacks = feedback_handler.dataset[feedback_handler.dataset["session_id"] == session_id]
    for _, entry in history.iterrows():
        st.session_state.messages.append({"id": entry["message_id"], "author": "user", "content": entry["message"]})
        fb = feedbacks[(feedbacks["response"] == entry["response"]) & (feedbacks["query"] == entry["message"])]
        if fb.empty:
            st.session_state.messages.append({"author": "assistant", "content": entry["response"]})
        else:
            last_feedback = fb.iloc[-1]
            st.session_state.messages.append({"id": last_feedback["message_id"],
                                              "author": "assistant",
                                              "content": entry["response"],
                                              "feedback": {"score": '👍' if last_feedback["is_satisfactory"] else '👎',
                                                           "text": last_feedback["comment"]}})


class Agent:
    def __init__(self):
        self.pipeline_obj = ModelPipeline()

    def process_document(self, uploaded_files):
        self.pipeline_obj.process_document(uploaded_files)

    def __call__(self, llm, params, query, enable_rag, session_id, enable_code_interpreter=False):
        res = self.pipeline_obj.generate_prediction(llm, params, query, enable_rag=enable_rag, session_id=session_id)
        response = res['prediction']
        return response.get('id'), response.get('answer'), response.get('source_documents')


def _submit_feedback(user_response, result, emoji=None):
    st.toast(f"Feedback submitted: {user_response} \n Conversation: {result}", icon=emoji)
    feedback_handler.add_row({
        "time": datetime.datetime.utcnow().isoformat(),
        "query": result["user"],
        "response": result["assistant"],
        "comment": user_response['text'],
        "is_satisfactory": user_response['score'] == "👍",
        "session_id": st.query_params.session_id,
        "message_id": result["id"]
    })
    return user_response.update(result)


def build_sources(sources):
    with res_placeholder:
        html = ""
        for r in sources:
            ctn = " ".join(r[0].split())[:512]
            ctn = ctn + f"... <a href=\"{r[1]}\" target=\"_blank\">See more.</a>"
            html += "<div class=\"row card\" style=\"border-radius: 3px; border-bottom: 1px solid #f2a9a2;margin-bottom: 10px; padding: 0px 10px 0px;\"><p style=\"font-family: Arial, sans-serif;\">" + ctn + "</p></div>"
        components.html(html, height=1000, scrolling=True)


def urlsafe_base64_encode(data: str) -> str:
    """
    Encodes a given string to URL-safe Base64.

    :param data: The string to encode.
    :return: The URL-safe Base64 encoded string.
    """
    # Convert the input string to bytes
    byte_data = data.encode('utf-8')

    # Encode the bytes to URL-safe Base64
    base64_bytes = base64.urlsafe_b64encode(byte_data)

    # Convert the Base64 bytes back to a string
    base64_string = base64_bytes.decode('utf-8')

    return base64_string


def _submit_session_form():
    session_id = urlsafe_base64_encode(f"{st.session_state.name}_{st.session_state.team}_{st.session_state.proficiency.lower()}")
    st.query_params.session_id = session_id
    if not session_handler.is_session_exist(session_id):
        session_handler.add_row({
            "created_on": datetime.datetime.utcnow().isoformat(),
            "session_id": session_id,
            "name": st.session_state.name,
            "team": st.session_state.team,
            "proficiency": st.session_state.proficiency
        })


def is_query_disabled(yes=False):
    return yes or any(msg is None for msg in
        [m.get("feedback", None) for m in st.session_state.messages if m["author"] == "assistant"])


@st.cache_resource
def load_agents():
    agent = Agent()
    return agent


if __name__ == "__main__":
    agent = load_agents()
    models = ["gemini_pro", "gemini_flash", "gemma", "tinyllama"]
    logger.info("starting app...")
    st.title("Sheet-Happens Chatbot")

    if "session_id" not in st.query_params or not session_handler.is_session_exist(st.query_params.session_id):
        with st.form("session_form"):
            name = st.text_input("Name", key="name")
            team = st.text_input("Company", key="team")
            proficiency = st.selectbox("RAG Proficiency", key="proficiency", options=("Beginner", "Intermediate", "Expert"))
            session_form_submit = st.form_submit_button("Begin", on_click=_submit_session_form)

    else:
        col1, col2, col3, col4 = st.columns(4)
        action1 = col1.button("What was the quarterly dividend declared per share in fiscal year 2023?")
        action2 = col2.button("How much was spent on share repurchases in fiscal year 2023?")
        action3 = col3.button("What was the total dividend per share for fiscal year 2023?")
        action4 = col4.button("Which segment generated the highest revenue in fiscal year 2023?")

        sidetab0, sidetab1, sidetab2 = st.sidebar.tabs(["File Upload", "Source Documents", "Settings"])

        with sidetab0:
            uploaded_files = st.file_uploader("Upload your PDF here", type=["pdf"], accept_multiple_files=True, key='uploaded_files')
            spinner_placeholder = st.empty()

        with sidetab1:
            res_placeholder = st.empty()

        with sidetab2:
            llm = st.radio("Set LLM", models, index=3, horizontal=True)
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.01, value=1.0)
            top_p = st.slider("TopP", min_value=0.0, max_value=1.0, step=0.01, value=1.0)
            max_output_tokens = st.number_input("MaxOutputTokens", step=1, value=2048)
            enable_rag = st.checkbox('Enable RAG', value=True)
            enable_code_interpreter = st.checkbox('Enable Code Interpreter', value=False)
            st.caption("Note: Enabling Code Interpreter will disable RAG and non-RAG agent.")

        if "uploaded_files_previous" not in st.session_state:
            st.session_state["uploaded_files_previous"] = []

        if "sources" not in st.session_state:
            st.session_state.sources = []

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if len(st.session_state.messages) == 0:  # only retrieve on start up
            get_session_history(st.query_params.session_id)

        # Display chat messages from history on app rerun
        for n, message in enumerate(st.session_state.messages):
            with st.chat_message(message["author"]):
                st.markdown(message["content"])

            if message["author"] == "assistant" and n >= 0:
                feedback_key = f"feedback_{int(n / 2)}"

                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = None

                if "feedback" not in message:
                    feedback = streamlit_feedback(
                        feedback_type="thumbs",
                        optional_text_label="Please provide extra information",
                        on_submit=_submit_feedback,
                        key=feedback_key,
                        kwargs={"result": {"id": st.session_state.messages[n-1]["id"], "user": st.session_state.messages[n-1]["content"], "assistant": message["content"]}}
                    )

            if "feedback" in message and message["feedback"]['text'] and message['feedback']['score'] == '👍':
                st.success(f"Comment: {message['feedback']['text']}")

            elif "feedback" in message and message["feedback"]['text'] and message['feedback']['score'] == '👎':
                st.error(f"Comment: {message['feedback']['text']}")

        build_sources(st.session_state.sources)

        # Accept user input

        if uploaded_files and uploaded_files != st.session_state["uploaded_files_previous"]:
            if len(uploaded_files) > len(st.session_state["uploaded_files_previous"]):
                with spinner_placeholder:
                    with st.spinner("Processing file..."):
                        agent.process_document([f for f in uploaded_files if f not in st.session_state["uploaded_files_previous"]])
                    st.info("File(s) added.")
            elif len(uploaded_files) < len(st.session_state["uploaded_files_previous"]):
                with spinner_placeholder:
                    st.info("File(s) removed.")

            st.session_state["uploaded_files_previous"] = uploaded_files

        res = []

        query = st.chat_input()

        if query or action1 or action2 or action3 or action4:

            if action1:
                query = "What was the quarterly dividend declared per share in fiscal year 2023?"
            elif action2:
                query = "How much was spent on share repurchases in fiscal year 2023?"
            elif action3:
                query = "What was the total dividend per share for fiscal year 2023?"
            elif action4:
                query = "Which segment generated the highest revenue in fiscal year 2023?"

            # Add user message to chat history
            st.session_state.messages.append({"author": "user", "content": query.replace('\'','')})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(query)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                # (self, return_references, query, task='instruct', target_lang: str='', llm: str='codey'):
                params = {"temperature": temperature, "max_output_tokens": max_output_tokens, "top_p": top_p}
                if enable_rag:
                    message_id, response, res = agent(llm=llm, params=params, query=query, enable_rag=True, session_id=st.query_params.session_id)
                else:
                    message_id, response, res = agent(llm=llm, params=params, query=query, enable_rag=False, session_id=st.query_params.session_id)


                # Display q&a response
                message_placeholder.markdown(response if response else "")

            if res:
                st.session_state.sources = res
                # build_sources(res)

            # Update user's message
            st.session_state.messages[-1]["id"] = message_id

            # Append q&a response
            st.session_state.messages.append({"id": message_id, "author": "assistant", "content": response})
            st.rerun()
