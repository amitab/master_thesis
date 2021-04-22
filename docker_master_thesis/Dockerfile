FROM jupyter/tensorflow-notebook

WORKDIR /home/jovyan/work

RUN pip install keras imgaug torch tensorflow_addons

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

USER root
RUN apt update
RUN apt install curl --assume-yes
USER $NB_UID

EXPOSE 8888

CMD ["jupyter", "lab", "--autoreload", "--NotebookApp.allow_remote_access=True", "--NotebookApp.token=''"]