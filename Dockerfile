FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu-py3

# install conda (?) (from https://raw.githubusercontent.com/ElementAI/docker-templates/master/Dockerfile-full-config-gpu?token=AELUSP_QvcsqTYn6fUxYWaituTsR9c08ks5aa6T4wA%3D%3D)
#RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion
#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && wget --no-check-certificate --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && /bin/bash /Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && rm Miniconda3-latest-Linux$
# my version
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && wget --no-check-certificate --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && rm Miniconda3-latest-Linux-x$
ENV PATH /opt/conda/bin:$PATH


# just copied from requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install six
RUN conda install mpi4py
RUN pip install cached_property
RUN pip install python-dateutil
RUN pip install tqdm
RUN pip install matplotlib

# install TF (again) to avoid conda issues
RUN pip install --ignore-installed --upgrade  https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp36-cp36m-linux_x86_64.whl

# for data
RUN pip install imageio

