FROM intelaipg/intel-optimized-tensorflow:latest-mkl-py3
LABEL maintainer="Glen Ferguson and Michoel Snow, all blame to the latter and glory to the former!"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
         build-essential \
         ca-certificates \
         cmake \
         curl \
         git \
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
WORKDIR /home/cordcomp/cord_comp
RUN chown cordcomp /home
RUN chown cordcomp /home/cordcomp
RUN chown cordcomp /home/cordcomp/cord_comp
USER cordcomp

COPY --chown=cordcomp:cordcomp *.py /home/cordcomp/cord_comp/
COPY --chown=cordcomp:cordcomp ./model_definitions/*.py /home/cordcomp/cord_comp/model_definitions/
COPY --chown=cordcomp:cordcomp config.gin /home/cordcomp/cord_comp/
COPY --chown=cordcomp:cordcomp requirements.txt /home/cordcomp/cord_comp/

RUN pip install opencv-python tqdm matplotlib scipy seaborn --user
RUN pip install /home/cordcomp/cord_comp/. --user
ENV PATH=$PATH:/home/cordcomp/.local/bin
ENTRYPOINT ["bash"]

