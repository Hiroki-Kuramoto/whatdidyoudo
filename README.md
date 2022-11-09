# whatdidyoudo

<img width="1080" alt="whatdidyoudo" src="https://user-images.githubusercontent.com/68832594/200877184-047a11f2-a0e5-4746-a310-c436759d3e9c.png">


- This project aims to infer the bug reproduction steps from Visual Contents.
- All you have to do is only to provide two images (e.g., images just before and after the issue occured).
- From buttons which is extracted from the former image with [Yolov5](https://github.com/ultralytics/yolov5/releases/tag/v6.2) object detection algorithm, this tool answer the most relevant one to the content of the latter image using [BERT](https://github.com/UKPLab/sentence-transformers).

## Damn! What did you do!
- Software developers always bash such words against the computer screen. Software developers are under severe stress. This is due to the bug reports we have written without consideration for the developers. With such content of report, software developers will have a hard time reproducing bugs because it is insufficient information for developers to reproduce the bug. 
**Please paste Visual Contents for the sake of it.** And leave the rest to me! 

## Requirements
- Need `git-lfs` to clone this repository.
- Need to prepare `Docker` environment.
  - We recommend the version `3.5.0` of Docker.

## How to execute
1. `cd <your clone>`
2. `sh exec.sh <former image path> <latter image path>`

## Recommended usages
- Two images just before and after the issue occurred.


## :copyright:Copyrights related
For object detection, we use the object detection algorithm [ultralytics/yolov5/v6.2](https://github.com/ultralytics/yolov5/releases/tag/v6.2).

For natural language processing, we use natural language processing algorithms [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers), especially pre-trained models [multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1).

For training our yoloSE model, annotated website screenshot [elsa-weber/buttondetection2](https://universe.roboflow.com/elsa-weber/buttondetection2) is used as a dataset.
