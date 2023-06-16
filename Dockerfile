FROM python:3.8-slim-buster
EXPOSE 8501
RUN apt update -y && apt install awscli -y

WORKDIR /app 

COPY . /app
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit" "run"]
CMD ["app.py"]