FROM qts8n/cuda-python:9.1-devel
LABEL author=DanielJunior email="danieljunior@id.uff.br"
ARG USER_ID
ARG GROUP_ID
ARG WANDB_API_KEY
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git gcc nano unzip

ENV PATH="/opt/miniconda3/bin:${PATH}"
ARG PATH="/opt/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh \
    && mkdir /opt/.conda \
    && bash Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -f Miniconda3-py310_24.1.2-0-Linux-x86_64.sh \
    && echo "Running $(conda --version)"

RUN mkdir -p /app
COPY . /app
WORKDIR /app

#RUN addgroup --gid $GROUP_ID user && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
#RUN chown -R $USER_ID:$GROUP_ID /app && chmod -R a=rwx /app && \
#    chown -R $USER_ID:$GROUP_ID /opt && chmod -R a=rwx /opt
#USER user

RUN conda init &&\
    conda update -n base conda &&\
    conda install -n base conda-libmamba-solver &&\
    conda config --set solver libmamba


RUN conda env create -f environment.yml -v

## Make RUN commands use the new environment:
RUN echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc &&\
    echo "conda activate deepal" >> ~/.bashrc
