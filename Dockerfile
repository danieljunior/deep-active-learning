FROM continuumio/miniconda3:22.11.1
LABEL author=DanielJunior email="danieljunior@id.uff.br"

RUN mkdir -p /app
COPY . /app
WORKDIR /app

RUN conda env create -f new_environment.yml -v
RUN conda update -n base conda &&\
    conda install -n base conda-libmamba-solver &&\
    conda config --set solver libmamba

# Make RUN commands use the new environment:
RUN echo "conda activate deepal" >> ~/.bashrc
