import argparse
import glob
# for settiongs
import os
import json
import configparser

# for Trimming
from PIL import Image
import numpy

# for images2text
import pyocr
import pyocr.builders

# for textCleaner
import re

# for textSimilarity
from sentence_transformers import SentenceTransformer, util

# for describe the bounding box
import cv2


# for remove working directory
import shutil

def main():
    """
     command line arguments
    """
    FORMER_IMG = args.former
    FORMER_EXTENSION = get_rightText(FORMER_IMG[-5:], ".")
    LATTER_IMG = args.latter
    LATTER_EXTENSION = get_rightText(LATTER_IMG[-5:], ".")
    SAVE_LOG = args.noSaveLog
    CANDIDATES = args.candidates
    SAVE_RESULT_TEXT = args.noSaveResultJson
    SAVE_RESULT_IMAGE = args.noSaveResultImage
    SAVE_RESULT = SAVE_RESULT_TEXT or SAVE_RESULT_IMAGE

    """
     settings
    """
    # read configurations
    config = configparser.ConfigParser()
    config.read('configurations.cfg')
    LOG_DIR = config['DEFAULT']['LOG_DIR'] # directory path
    RESULT_DIR = config['DEFAULT']['RESULT_DIR'] # directory path
    WORKING_DIR = "output/work" # working directory path
    YOLOSE_CLASS = config['DEFAULT']['YOLOSE_CLASS'] # json
    YOLOSE_RESULT = config['DEFAULT']['YOLOSE_RESULT'] # directory path
    BERT_MODEL = config['DEFAULT']['BERT_MODEL'] # bert model name

    # create working directory for this execution
    count = 2
    # check if the working directory exists
    while os.path.exists(f"{YOLOSE_RESULT}{str(count)}"):
        count += 1
    if count > 2:
        WORKING_DIR  += str(count-1)
        LOG_DIR      += str(count-1)
        RESULT_DIR   += str(count-1)
        YOLOSE_RESULT+= str(count-1)
    os.mkdir(WORKING_DIR)
    os.mkdir(WORKING_DIR + "/images")
    os.mkdir(WORKING_DIR + "/ocr_output")
    os.mkdir(WORKING_DIR + "/image_identity")
    # create log/result directory
    if SAVE_LOG and (not os.path.exists(LOG_DIR)):
        os.mkdir(LOG_DIR)
        os.mkdir(LOG_DIR + "/images")
        os.mkdir(LOG_DIR + "/ocr_output")
        os.mkdir(LOG_DIR + "/image_identity")
    if SAVE_RESULT and (not os.path.exists(RESULT_DIR)):
        os.mkdir(RESULT_DIR)

    # get yoloSE classes
    assert os.path.exists(YOLOSE_CLASS), "whatdidyoudo >>\033[33m YOLOSE_CLASS file does not exist.\033[0m"
    with open(YOLOSE_CLASS, 'r') as f:
        yoloSE_classes = json.load(f)
    for yoloSE_class in yoloSE_classes.values():
        if not os.path.exists(f"{WORKING_DIR}/images/{yoloSE_class}"):
            os.mkdir(f"{WORKING_DIR}/images/{yoloSE_class}")
            os.mkdir(f"{WORKING_DIR}/ocr_output/{yoloSE_class}")
            os.mkdir(f"{WORKING_DIR}/image_identity/{yoloSE_class}")
        if SAVE_LOG and (not os.path.exists(f"{LOG_DIR}/images/{yoloSE_class}")):
            os.mkdir(f"{LOG_DIR}/images/{yoloSE_class}")
            os.mkdir(f"{LOG_DIR}/ocr_output/{yoloSE_class}")
            os.mkdir(f"{LOG_DIR}/image_identity/{yoloSE_class}")
        

    """
     trimming the button locations detected by yoloSE
    """
    image = Image.open(FORMER_IMG)
    with open(f"{YOLOSE_RESULT}/labels/former.txt", "r") as f:
        for i, line in enumerate(f):
            line = line.split()
            # get the image information from yoloSE results
            image_class = yoloSE_classes[str(line[0])]
            center_x, center_y, width, height = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            center_x, center_y, width, height = center_x * image.width, center_y * image.height, width * image.width, height * image.height
            # trimming
            trimmed = image.crop((center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2))
            
            # store the trimmed image
            path = f"/images/{image_class}/{str(i)}.{FORMER_EXTENSION}"
            trimmed.save(f"{WORKING_DIR}{path}")
            if SAVE_LOG:
                trimmed.save(f"{LOG_DIR}{path}")

            # save the image identity
            with open(f"{WORKING_DIR}/image_identity/{image_class}/{str(i)}.json", "w") as f:
                json.dump({"center_x": center_x, "center_y": center_y, "width": width, "height": height, "topLeft": [center_x-width/2, center_y-height/2], "downRight": [center_x+width/2, center_y+height/2]}, f)
            if SAVE_LOG:
                with open(f"{LOG_DIR}/image_identity/{image_class}/{str(i)}.json", "w") as f:
                    json.dump({"center_x": center_x, "center_y": center_y, "width": width, "height": height, "topLeft": [center_x-width/2, center_y-height/2], "downRight": [center_x+width/2, center_y+height/2]}, f)

    del image, trimmed

    """
     using OCR, extract documents from the trimmed images
    """
    # get the tools
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        raise Exception("whatdidyoudo >>\033[33m No OCR tool found\033[0m")
    tool = tools[0]
    print(f"whatdidyoudo >> Will use tool '{tool.get_name()}'")
    langs = tool.get_available_languages()
    if "eng" not in langs:
        raise Exception("whatdidyoudo >>\033[33m The OCR tool does not support English\033[0m")
    print(f"whatdidyoudo >> Available languages: {langs}")
    lang = "eng"
    print(f"whatdidyoudo >> Will use lang '{lang}'")

    for yoloSE_class in yoloSE_classes.values():
        for image_path in glob.glob(f"{WORKING_DIR}/images/{yoloSE_class}/*.{FORMER_EXTENSION}"):
            # get the text
            doc = tool.image_to_string(
                Image.open(image_path),
                lang=lang,
                builder=pyocr.builders.TextBuilder()
            ).lower()
            doc = textCleaner(doc)
            # store the text
            path = f"{get_rightText(image_path, '/images/')[:-4]}.txt"
            with open(f"{WORKING_DIR}/ocr_output/{path}", "w") as f:
                f.write(doc)
            if SAVE_LOG:
                with open(f"{LOG_DIR}/ocr_output/{path}", "w") as f:
                    f.write(doc)

    """
     using OCR, extract texts from both images.
     and create a diff between the former and latter texts.
    """
    former_text = tool.image_to_string(
        Image.open(FORMER_IMG),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
        ).lower()
    former_text = textCleaner(former_text)
    if len(former_text) == 0:
        print("whatdidyoudo >>\033[33m Warning: the former_text is empty\033[0m")

    latter_text = tool.image_to_string(
        Image.open(LATTER_IMG),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
        ).lower()
    latter_text = textCleaner(latter_text)
    if len(latter_text) == 0:
        print("whatdidyoudo >>\033[33m Warning: the latter_text is empty\033[0m")

    diff = " " + latter_text + " "
    for former_token in former_text.split():
        diff = diff.replace(" " + former_token + " ", " ", 1)
    diff = diff.strip()

    # store the query, former text and latter text
    with open(f"{WORKING_DIR}/ocr_output/former.txt", "w") as f:
        f.write(former_text)
    with open(f"{WORKING_DIR}/ocr_output/latter.txt", "w") as f:
        f.write(latter_text)
    with open(f"{WORKING_DIR}/ocr_output/diff.txt", "w") as f:
        f.write(diff)
    if SAVE_LOG:
        with open(f"{LOG_DIR}/ocr_output/former.txt", "w") as f:
            f.write(former_text)
        with open(f"{LOG_DIR}/ocr_output/latter.txt", "w") as f:
            f.write(latter_text)
        with open(f"{LOG_DIR}/ocr_output/diff.txt", "w") as f:
            f.write(diff)

    # del former_text, latter_text

    """
     using text similarity, find the most similar documents to the query
    """
    # get the documents and selfcheck
    docs = {}
    for yoloSE_class in yoloSE_classes.values():
        for doc_path in glob.glob(f"{WORKING_DIR}/ocr_output/{yoloSE_class}/*.txt"):
            with open(doc_path, "r") as f:
                doc = f.read()
            assert type(doc) == str, f"whatdidyoudo >>\033[33m The document is not a string > {doc_path}\033[0m"
            if len(doc) == 0:
                print(f"whatdidyoudo >>\033[33m Warning: The document is empty > {doc_path}\033[0m")
            if len(doc.split()) > 512:
                raise ValueError(f"whatdidyoudo >>\033[33m Too long document (> 512 words). The document contains apploximately {len(doc.split())} tokens > {doc_path}\033[0m")
            docs[doc_path] = {"text":doc, "similarity": None}
    
    # Load the model
    try:
        print(f"whatdidyoudo >> Loading the BERT pre-trained model...{BERT_MODEL}")
        model = SentenceTransformer(BERT_MODEL)
    except:
        raise Exception("whatdidyoudo >>\033[33m BERT_MODEL is not found\033[0m")
    
    # Encode the query which imply the latter text minus the former text
    query_emb_sub = model.encode(latter_text) - model.encode(former_text)
    query_emb_diff = model.encode(diff)
    query_emb = query_emb_sub + query_emb_diff
    print(f"whatdidyoudo >> The query is encoded")

    # Encode the documents
    print(f"whatdidyoudo >> Encoding the documents...")
    for doc_path, doc in docs.items():
        doc_emb = model.encode(doc["text"])
        # Compute the cosine similarity
        doc["similarity"] = util.pytorch_cos_sim(query_emb, doc_emb)[0][0].item()
    print(f"whatdidyoudo >> The cosine similarities are computed.")
    
    # Sort the documents by their similarity to the query
    print(f"whatdidyoudo >> Sorting the documents by their similarity to the query...")
    BERT_output = {k: v for k, v in sorted(docs.items(), key=lambda item: item[1]["similarity"], reverse=True)}

    
    # print the results
    print(f"whatdidyoudo >> printing the top {CANDIDATES} buttons.")
    print_maxCount = CANDIDATES
    img = cv2.imread(FORMER_IMG)
    for doc_path, doc in BERT_output.items():
        if print_maxCount == 0:
            break
        else:
            print_maxCount -= 1
        print(f"whatdidyoudo >>\033[1m similarity:{doc['similarity']:5f} <{doc['text'][:15]:15}> @{doc_path}\033[0m")

        # hightlite the button on FORMER_IMG
        identity_path = doc_path.replace("/ocr_output/", "/image_identity/").replace(".txt", ".json")
        with open(identity_path, "r") as f:
            identity = json.load(f)
        topLeft   = tuple(map(int, identity["topLeft"]))
        downRight = tuple(map(int, identity["downRight"]))
        cv2.rectangle(img=img, pt1=topLeft, pt2=downRight, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_4)
        cv2.putText(img=img, text=f"{doc['similarity']:5f}", org=topLeft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(50, 50, 255), thickness=3, lineType=cv2.LINE_4)

    # save or show the image
    if SAVE_RESULT_IMAGE:
        cv2.imwrite(f"{RESULT_DIR}/result.{FORMER_EXTENSION}", img)

    # store the results in a json file
    with open(f"{RESULT_DIR}/results.json", "w") as f:
        json.dump(BERT_output, f)

    # remove the working directory
    shutil.rmtree(WORKING_DIR)



def textCleaner(text:str)->str:
    text = re.sub("\W", " ", text)
    text = re.sub("\d+", "0", text)
    text = text.split()
    text = " ".join(text)
    return text

def get_rightText(text, prefix):
    idx = text.find(prefix)
    t = text[idx+len(prefix):]
    return t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('former', type=str, help='former image path')
    parser.add_argument('latter', type=str, help='latter image path')
    parser.add_argument('--candidates', type=int, default=3, help='number of candidates to display in the resulting image (default: 3)') 
    parser.add_argument('--noSaveLog', action='store_false', help='do not save the logs')
    parser.add_argument('--noSaveResultJson', action='store_false', help='do not save the results as json')
    parser.add_argument('--noSaveResultImage', action='store_false', help='do not save the result image')
    
    args = parser.parse_args()
    assert args.candidates > 0, "whatdidyoudo >>\033[33m candidates must be positive\033[0m"
    main()