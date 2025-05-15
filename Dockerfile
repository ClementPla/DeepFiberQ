# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11-slim


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
USER root
# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip install light-the-torch && ltt install --pytorch-computation-backend=cu121 torch torchvision

# Invalidate the cache from this point on if the library version changes
ADD https://api.github.com/repos/ClementPla/DeepFiberQ/git/refs/heads/relabelled version.json
RUN git clone -b relabelled https://github.com/ClementPla/DeepFiberQ.git
WORKDIR "DeepFiberQ/"
RUN pip install .
EXPOSE 8501

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT ["streamlit", "run", "ui/Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]