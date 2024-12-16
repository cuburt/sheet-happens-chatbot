import datetime
import pandas as pd
from  pandas.errors import EmptyDataError
from pathlib import Path
import os
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent)


class Session:
    def __init__(self):
        # Initialize an in-memory dataset using a Pandas DataFrame
        self.filepath = os.path.join(PROJECT_ROOT_DIR, "data/session/chat_history.csv")
        self.dataset = self.load_from_csv()

    def load_from_csv(self):
        try:
            return pd.read_csv(self.filepath)
        except (FileNotFoundError, EmptyDataError):
            return pd.DataFrame(columns=["created_on", "message_id", "session_id", "message", "response"])

    def is_conversation_exist(self, session_id):
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

    def add_to_conversations(self, message_id, session_id, message=None, response=None):
        """
        Add a new conversation record to the dataset.
        """
        new_row = {
            "created_on": datetime.datetime.utcnow().isoformat(),
            "message_id": message_id,
            "session_id": session_id,
            "message": message,
            "response": response,
        }
        self.dataset = pd.concat([self.dataset, pd.DataFrame([new_row])], ignore_index=True)
        self.save_to_csv()
        return session_id

    def save_to_csv(self):
        """
        Save the dataset to a CSV file.
        """
        self.dataset.to_csv(self.filepath, index=False)
