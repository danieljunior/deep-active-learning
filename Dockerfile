FROM continuumio/miniconda3:22.11.1
LABEL author=DanielJunior email="danieljunior@id.uff.br"

RUN mkdir -p /app
COPY . /app
WORKDIR /app

RUN conda env create -f environment.yml -v
# Make RUN commands use the new environment:
RUN echo "conda activate deepal" >> ~/.bashrc
