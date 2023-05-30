FROM ros:humble

RUN apt update -y
RUN apt upgrade -y

RUN apt install -y python3.10

RUN apt install -y python3-pip

COPY camera /opt/app

WORKDIR /opt/app

RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["ssh","pi@192.168.1.222"]
CMD ["cd","project"]
CMD ["env/bin/activate"]
CMD ["python3","cam.py"]
