FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y \
  python-pip \
  python-dev \
  build-essential

COPY . /web

WORKDIR /web

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["server.py"]
