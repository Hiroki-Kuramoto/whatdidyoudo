#!/bin/bash

echo "\033[32m -*-*-*-*-*-*- copy your images in this directory -*-*-*-*-*-*- \033[0m"
mkdir src/input
mkdir src/input/images

cp $1 src/input/images/former.${1##*.}
cp $2 src/input/images/latter.${2##*.}


echo "\033[32m -*-*-*-*-*-*- create docker container -*-*-*-*-*-*- \033[0m"
docker-compose build
docker images |grep python3.9
docker-compose up -d
docker-compose ps


echo "\033[32m -*-*-*-*-*-*- execute command in container -*-*-*-*-*-*- \033[0m"
docker exec -it python3.9 python --version
docker exec -it python3.9 apt-get install -y tesseract-ocr
docker exec -it python3.9 python yolov5-6.2/detect.py --source input/images/former.${1##*.} --weights model/weights/yoloSEv5x.pt --name yolo-result --save-txt
docker exec -it python3.9 python whatdidyoudo_img.py input/images/former.${1##*.} input/images/latter.${2##*.}


echo "\033[32m -*-*-*-*-*-*- cleaning -*-*-*-*-*-*- \033[0m"
# docker exec -it python3.9 rm -rf output/


echo "\033[32m -*-*-*-*-*-*- stop container -*-*-*-*-*-*- \033[0m"
docker-compose down
docker-compose ps