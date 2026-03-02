FROM python:3.10-slim

# HuggingFace requires a non-root user with ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY  --chown=user README.md Dockerfile requirements.txt *.py $HOME/app/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
