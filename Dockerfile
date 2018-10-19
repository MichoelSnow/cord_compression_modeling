FROM intelaipg/intel-optimized-tensorflow:latest-mkl-py3

# Built off the intel optimized TF docker image
LABEL maintainer="Glen Ferguson and Michoel Snow, all blame to the latter and glory to the former!"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
         build-essential \
         ca-certificates \
         cmake \
         curl \
         git \
         libjpeg-dev \
         libpng-dev \
         python-qt4 \
         sudo \
         unzip \
         vim \
         wget \
          zip &&\
     rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash cordcomp
RUN usermod -aG sudo cordcomp
WORKDIR /opt
RUN chown cordcomp /opt
USER cordcomp
WORKDIR /home/cordcomp/cord_comp

COPY *.py /home/cordcomp/cord_comp
COPY ./model_definitions/*.py /home/cordcomp/cord_comp
COPY config.gin /home/cordcomp/cord_comp

ENV PYTHON_VERSION=3.6
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build

RUN /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
RUN conda install -y jupyter opencv bcolz tqdm matplotlib scipy seaborn graphviz python-graphviz keras pandas
RUN pip install sklearn-pandas isoweek pandas-summary gin-config
RUN pip install /home/cordcomp/cord_comp/setup.py
ENTRYPOINT ["bash"]
