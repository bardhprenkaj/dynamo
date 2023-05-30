FROM ubuntu
ARG DEBIAN_FRONTEND=noninteractive

ADD os_requirements.txt .
RUN apt update -y && apt install -y $(cat os_requirements.txt|grep -v '#')

# Install Python packages (from requirements.txt):

ADD requirements.txt .
RUN pip install -r requirements.txt


ARG UNAME=user
ARG GID=1000
ARG UID=1000
RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

RUN chmod -R 777 /home
USER $UNAME