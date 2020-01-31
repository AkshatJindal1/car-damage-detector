FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3-pip

WORKDIR /app

ADD custom_multiclass.py custom.py app.py setup.py requirements.txt /app/
ADD logs /app/logs/
ADD mrcnn /app/mrcnn/

RUN pip3 install -r requirements.txt && python3 setup.py install


ENV PYTHONPATH=/app

EXPOSE 5005

CMD [ "python3", "app.py"]