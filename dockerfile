FROM python:3.9
USER root

COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r /requirements.txt