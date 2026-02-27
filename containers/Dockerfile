FROM python:3.12-alpine

RUN apk add --no-cache git

ARG PIX=/opt/pixel_viewer
RUN git clone -b main https://github.com/cuddandr/larnd_pixel_viewer ${PIX}
RUN pip install --no-cache-dir --upgrade -r ${PIX}/requirements.txt

EXPOSE 8050

WORKDIR ${PIX}
CMD ["python3", "pixel_viewer.py", "--host", "0.0.0.0"]
