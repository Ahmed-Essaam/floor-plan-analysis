### 1. Get Linux
#FROM alpine:3.18
FROM openjdk:17-jdk-alpine

### 2. Get Java via the package manager
RUN apk update \
&& apk upgrade \
&& apk add --no-cache bash \
&& apk add --no-cache --virtual=build-dependencies unzip \
&& apk add --no-cache curl \
&& apk add --no-cache py3-pandas

### 3. Get Python, PIP

RUN apk add --no-cache python3=3.9.7-r3 \
&& python3 -m ensurepip \
&& pip3 install --upgrade pip setuptools \
&& rm -r /usr/lib/python*/ensurepip && \
if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 /usr/bin/python; fi && \
rm -r /root/.cache


# Set fallback mount directory
ENV MNT_DIR /mnt/gcs

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

###Create mount directory for service
RUN mkdir -p $MNT_DIR


### Get Flask for the app
RUN pip install --trusted-host pypi.python.org flask
RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev

COPY ./requirements.txt .
RUN pip install -r requirements.txt


EXPOSE 8081    
ADD handler.py /
CMD ["python", "handler.py"]