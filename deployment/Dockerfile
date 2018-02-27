FROM continuumio/miniconda3
RUN mkdir /opt/notebooks 

# Librarys
RUN conda install -c conda-forge -c ioam boto3=1.5.36 --yes
RUN conda install -c conda-forge -c iris=2.0.0 bokeh=0.12.13 cartopy=0.16.0 --yes
RUN conda install -c conda-forge pyke=1.1.1 --yes

# pysssix stuff
RUN apt-get update -y
RUN apt-get install libfuse-dev --fix-missing -y
RUN pip install git+git://github.com/met-office-lab/pysssix.git

# TODO use a differnet user rather than root.

RUN apt-get update
RUN apt-get install libgl1-mesa-swx11 -y

ENV PATH="/opt/conda/bin/:${PATH}"
ENV PYTHONPATH=$PYTHONPATH:/root/.jupyter/extentions:/opt/forest_lib

WORKDIR /root

COPY run.sh /root
CMD ["/root/run.sh"]
EXPOSE 8888