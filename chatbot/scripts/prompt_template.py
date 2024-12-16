from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class IQPromptTemplate:
    def __init__(self, response_format: str = ""):
        template_string = """
You are a financial data processing assistant. Your task is to extract, analyze, and summarize structured financial \
information from a document. The document contains tabular data such as balance sheets, income statements, and other \
financial metrics. Ensure the data is extracted accurately and presented in a clean, organized format.

Your task is to:
1. Extract tabular data, ensuring all rows, columns, and headers are correctly aligned.
2. Identify key financial metrics such as Revenue, Expenses, and Net Profit.
3. Summarize the financial performance trends across different time periods.
4. Highlight any noteworthy observations or anomalies, such as sudden revenue growth or expense spikes.
5. Provide the extracted table in JSON format for further processing.


Chat History:
{chat_history}

Question:
{input}

Context:
{context}

{format_instructions}

Helpful AI Response:
"""

        self.prompt = PromptTemplate(
            template=template_string,
            input_variables=["context", "input", "chat_history"],
            partial_variables={"format_instructions": f"format Instructions:\n{response_format}"}
        )


class DocumentParserPromptTemplate:
    def __init__(self):
        template = "page_content: {page_content}\nsource: {source}"

        self.prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template=template
        )
