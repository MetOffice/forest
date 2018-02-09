FROM continuumio/miniconda3
RUN mkdir /opt/notebooks 

# Librarys
RUN conda install -c conda-forge -c ioam boto3 iris bokeh cartopy -y
RUN conda install -c conda-forge pyke -y

# pysssix stuff
RUN apt-get update -y
RUN apt-get install libfuse-dev --fix-missing -y
RUN pip install git+git://github.com/met-office-lab/pysssix.git
RUN mkdir /s3

RUN apt-get install libfuse-dev --fix-missing -y
RUN wget -O goofys https://github.com/kahing/goofys/releases/download/v0.19.0/goofys
RUN chmod u+x goofys
RUN mkdir -p /root/bin
RUN mv goofys /root/bin
ENV PATH="/root/bin/:${PATH}"
RUN mkdir /s3


# TODO use a differnet user rather than root.

RUN apt-get update
RUN apt-get install libgl1-mesa-swx11 -y

ENV PATH="/opt/conda/bin/:${PATH}"
ENV PYTHONPATH=$PYTHONPATH:/root/.jupyter/extentions:/opt/forest_lib

WORKDIR /root

COPY run.sh /root
CMD ["/root/run.sh"]
EXPOSE 8888