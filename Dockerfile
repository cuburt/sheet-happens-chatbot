FROM python:3.10

# Set the working directory to /streamlit-app
RUN mkdir /streamlit-app
WORKDIR /streamlit-app

# copy the requirements file used for dependencies
RUN mkdir streamlit_feedback
RUN mkdir streamlit_feedback/frontend
RUN mkdir streamlit_feedback/frontend/public
RUN mkdir streamlit_feedback/frontend/src
COPY streamlit_feedback/__init__.py streamlit_feedback/
COPY streamlit_feedback/frontend/package.json streamlit_feedback/frontend/
COPY streamlit_feedback/frontend/package-lock.json streamlit_feedback/frontend/
COPY streamlit_feedback/frontend/tsconfig.json streamlit_feedback/frontend/
COPY streamlit_feedback/frontend/public/ streamlit_feedback/frontend/public/
COPY streamlit_feedback/frontend/src/ streamlit_feedback/frontend/src/
COPY . .

RUN apt-get -y update; apt-get -y install \
  curl \
  jq \
  ca-certificates \
  nano \
  npm \
  tesseract-ocr \
  libtesseract-dev \
  poppler-utils

RUN cd streamlit_feedback/frontend && npm install react-scripts --save
RUN cd streamlit_feedback/frontend && npm run build

ARG GEMINI_APIKEY
ENV GEMINI_APIKEY=${GEMINI_APIKEY}
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install any needed packages specified in requirements.txt
RUN pip3 install streamlit pandas
RUN pip3 install -r requirements.txt
EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
