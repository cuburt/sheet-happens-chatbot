import os
from pathlib import Path
import pandas as pd
from  pandas.errors import EmptyDataError
from log import logger

PROJECT_ROOT_DIR = str(Path(__file__).parent)


class DatasetHandler:
    def __init__(self, filepath, columns):
        # Initialize an in-memory dataset using a Pandas DataFrame
        self.filepath = os.path.join(PROJECT_ROOT_DIR, filepath)
        if not os.path.exists(os.path.dirname(self.filepath)):
            os.makedirs(os.path.dirname(self.filepath))
            logger.info(f"csv file created at {self.filepath}")
        self.columns = columns
        self.dataset = self.load_from_csv()

    def load_from_csv(self):
        try:
            return pd.read_csv(self.filepath)
        except (FileNotFoundError, EmptyDataError):
            return pd.DataFrame(columns=self.columns)

    def is_session_exist(self, session_id):
        """
        Check if a conversation exists for the given session_id.
        """
        return not self.dataset[self.dataset["session_id"] == session_id].empty

    def get_conversations(self, session_id, memory_window):
        """
        Retrieve conversations for a session_id, sorted by created_on, limited by memory_window.
        """
        filtered = self.dataset[self.dataset["session_id"] == session_id]
        sorted_data = filtered.sort_values(by="created_on", ascending=False)
        return sorted_data.head(memory_window)

    def get_session_history(self, session_id):
        return self.dataset[self.dataset["session_id"] == session_id].sort_values(by="created_on")

    def add_row(self, row):
        self.dataset = pd.concat([self.dataset, pd.DataFrame([row])], ignore_index=True)
        self.save_to_csv()

    def save_to_csv(self):
        """
        Save the dataset to a CSV file.
        """
        self.dataset.to_csv(self.filepath, index=False)

    def query(self, filter_func):
        return self.dataset[filter_func(self.dataset)]


feedback_handler = DatasetHandler(
    "data/session/feedback.csv",
    ["time", "query", "response", "comment", "is_satisfactory", "session_id", "message_id"]
)
session_handler = DatasetHandler(
    "data/session/session.csv",
    ["created_on", "session_id", "name", "team", "proficiency"]
)
history_handler = DatasetHandler(
    "data/session/chat_history.csv",
    ["created_on", "message_id", "session_id", "message", "response"]
)
