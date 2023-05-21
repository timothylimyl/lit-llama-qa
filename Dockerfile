FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


# Install python3
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -yq update \
    && apt-get install -y --no-install-recommends python3-dev python3-pip git

# Add in dependencies here:
WORKDIR /code
COPY ./requirements.txt /code
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt autoremove  