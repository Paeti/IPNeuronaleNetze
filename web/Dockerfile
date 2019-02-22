FROM ubuntu:18.04

MAINTAINER Patrick Reckeweg
RUN apt-get update -y && apt-get install -y \
  python-pip \
  python-dev \
  build-essential

COPY . /ip/app

WORKDIR /ip/app

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["server.py"]