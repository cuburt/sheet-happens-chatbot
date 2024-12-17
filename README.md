# Retrieval Augmented Chatbot For Tabular Data

### Setup:
- run pip install -r requirements
- run streamlit run app.py 
- you can now access the app at http://localhost:8080 
- to reset session, reenter http://localhost:8080 without session id in your browser.
- Also deployed at https://sheet-happens-137134699526.us-central1.run.app

*Note: To reset conversation, just create another session by changing some details in the credential form. To go to the form, open this link https://sheet-happens-137134699526.us-central1.run.app without session id*

### Design
#### Document Preprocessing and Indexing
 - UnstructuredPDFLoader from Langchain.document_loaders is used to read the uploaded PDF files.
 - NLTKTextSplitter for chunking the document with chunk size of 512 and overlap of 5.
 - FAISS is used for indexing and in-memory vector storage.
 - sentence-transformers/all-MiniLM-L6-v2 for embedding the documents.
 - Multithreading is utilised during indexing and preprocessing to improve concurrency in cpu environment and keep the server alive during such phase.
#### Retrieval
 - Maximal marginal relevance search method is used to keep retrieved documents diverse yet relevant.
 - cross-encoder/ms-marco-MiniLM-L-6-v2 for reranking
 - LongContextReorder is used to reorder the retrieved documents and increase LLM's comprehension of long contexts.
#### Generation
 - A customised LLMChain for Gemini Flash API is used.
 - Inference parameters such as temperature and top p are adjustable from the user interface.
 - A customised retrieval chain is also used with adjustable memory window of 3 and configurable LLM parameters.
 - Gemini Pro and Gemma 2 are also implemented but currently limited.
#### Conversation Management
 - Multiple sessions each with its own saved history is supported.
 - Feedback is also supported.
 - All data is saved as a dataset in the local directory.

### Challenges
In RAG systems, many components are working together. So there are many components that can fail and can be improved on.  Its a challenge to find the optimal combination. A different loader or chunking method might work better or bigger encoder might be better, or it could be the LLM that can be worked on.
Prompt engineering is also a challenge. since LLMs are generative, its hard to get a consistent result even with controllable inference parameters. for now, a simple prompt template is implemented. 

### Solutions
By implementing all this in Langchain, it is very easy to customise however I like. A lot of Langchain objects I used are customised based on how I see fit. The components are wrapped and can be easily orchestrated into a one single pipeline.

### Prompt Engineering Strategies
A prompt template is used to structure the query, chat history, contexts, and instruction.
the first part of the template is a premise to narrow down the LLM's parametric knowledge to or domain usecase:

*"You are a Financial Data Processing Assistant. Your role is to extract, analyze, and summarize structured or 
unstructured financial information from a PDF document. The document may contain tabular data such as balance sheets, 
income statements, and other financial metrics."*

followed by a step-by-step main instruction:

*"Based on the user's question, perform the following:*
*1. Respond accurately to any specific questions about the PDF file's financial data.*
*2. If applicable, extract relevant tables from the PDF, ensuring all rows, columns, and headers are correctly aligned.*
*3. Identify key financial metrics such as Revenue, Expenses, and Net Profit.*
*4. Summarize the financial performance trends across different time periods.*
*5. Highlight any noteworthy observations or anomalies, such as sudden revenue growth or expense spikes.*
*6. Provide the extracted table in JSON format for further processing."*

Then the user's query and chat history are appended.

a format instruction can be included to force the model to generate the response in a certain way like in JSON.

