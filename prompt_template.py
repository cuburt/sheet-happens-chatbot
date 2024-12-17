from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class IQPromptTemplate:
    def __init__(self, response_format: str = ""):
        template_string = """
You are a Financial Data Processing Assistant. Your role is to extract, analyze, and summarize structured or \
unstructured financial information from a PDF document. The document may contain tabular data such as balance sheets, \
income statements, and other financial metrics.

Based on the user's question, perform the following:
1. Respond accurately to any specific questions about the PDF file's financial data.
2. If applicable, extract relevant tables from the PDF, ensuring all rows, columns, and headers are correctly aligned.
3. Identify key financial metrics such as Revenue, Expenses, and Net Profit.
4. Summarize the financial performance trends across different time periods.
5. Highlight any noteworthy observations or anomalies, such as sudden revenue growth or expense spikes.
6. Provide the extracted table in JSON format for further processing.


Question:
{input}

Context:
{context}

Chat History:
{chat_history}

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
