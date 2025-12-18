FROM public.ecr.aws/emr-serverless/spark/emr-7.11.0:latest

USER root

# Install Python deps into the image so every driver/executor has them
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

# EMR Serverless expects to run as hadoop
USER hadoop:hadoop
