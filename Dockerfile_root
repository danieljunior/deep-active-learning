FROM qts8n/cuda-python:9.1-devel
LABEL author=DanielJunior email="danieljunior@id.uff.br"
USER root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git gcc nano unzip

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda

RUN mkdir -p /app
COPY . /app
WORKDIR /app
COPY netrc /root/.netrc

RUN conda update -n base conda &&\
    conda install -n base conda-libmamba-solver &&\
    conda config --set solver libmamba

RUN conda env create -f environment.yml -v

# Make RUN commands use the new environment:
RUN echo "conda activate deepal" >> ~/.bashrc
