# works with tensorflow 2.9.1
#FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 
# base (also works with tensorflow 2.9.1)
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04  

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	build-essential ca-certificates python3.8 python3.8-dev python3.8-distutils git wget cmake
RUN ln -sv /usr/bin/python3.8 /usr/bin/python

# Create a non-root user
RUN groupadd -g $USER_GID $USERNAME \
	&& useradd -u $USER_UID -g $USER_GID -m $USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py --user && \
	rm get-pip.py

# install dependencies
RUN pip install --user opencv-python
RUN pip install --user matplotlib
RUN pip install --user albumentations
RUN pip install --user tensorboard
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user torch==2.0.1 torchvision torchaudio
RUN pip install --user torch-tb-profiler
RUN pip install --user neptune-client
RUN pip install --user optuna
# additions for registration
RUN pip install --user numpy==1.23.5
RUN pip install --user spectral
RUN pip install --user scikit-learn
RUN pip install --user pystackreg
RUN pip install --user glob2


RUN pip install --user onnx 
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --user -e detectron2_repo
# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# install linters
RUN pip install --user flake8
RUN pip install --user pylint

# install the Neptune-Optuna integration
RUN pip install --user neptune-optuna

RUN pip install --user fiftyone
RUN pip install --user shapely
RUN pip install --user torcheval
RUN pip install --user tqdm
RUN pip install --user hydra-core
RUN pip install --user scipy
RUN pip install --user imantics
RUN pip install --user pandas
RUN pip install --user xlrd
RUN pip install --user openpyxl
RUN pip install --user seaborn
RUN pip install --user SimpleITK
RUN pip install --user scikit-image
# bayesian optimization
RUN pip install --user scikit-optimize
RUN pip install --user bayesian-optimization
RUN pip install --user mlxtend
#RUN pip install --user tensorflow_addons


# tensorflow
#RUN pip install --user --upgrade pip
#RUN pip install --user tensorflow==2.9.1


CMD ["bash"]
WORKDIR /exp