# API Specific
from flask import Flask, request, make_response, jsonify
import json
import flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

#Model Specific
import numpy
import numpy as np
import cv2
import custom
import custom_multiclass
from mrcnn import model as modellib, utils
from matplotlib import patches,  lines
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from mrcnn.config import Config
import skimage
from mrcnn import visualize
import imageio
from PIL import Image
import io
import base64

#GCP specific
import uuid
from google.cloud import storage
from tempfile import NamedTemporaryFile

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'

client = storage.Client().create_anonymous_client()
bucket = client.get_bucket('damaged-car-images')


app = Flask(__name__)
ROOT_DIR = os.getcwd()

config_damage = custom.CustomConfig()
config_parts = custom_multiclass.CustomConfig()
MODEL_DIR = os.path.join(ROOT_DIR, "logs/")

parts = {
    1: "rear_bumper",
    2: "front_bumper",
    3: "headlamp",
    4: "door",
    5: "hood"
}

class InferenceConfigDamage(config_damage.__class__):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

class InferenceConfigParts(config_parts.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config_damage = InferenceConfigDamage()
config_parts= InferenceConfigParts()

def display_instances(image, boxes, masks, class_ids):
    masked_image = image.copy()
    # Number of instances
    N = boxes.shape[0]
    if not N:
        return masked_image
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    # Generate random colors
    colors = visualize.random_colors(N)

    for i in range(N):
        color = colors[i]
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, color)
    return masked_image

def andOperation(part,damage):
    return np.logical_and(part,damage)

def map_part_to_damage(damage, parts):
    if(damage.shape[2]==0 or parts.shape[2]==0):
        return np.zeros((parts.shape[-1],parts.shape[0], parts.shape[1])) 
    damage = (np.sum(damage, -1, keepdims=True) >= 1)
    return np.array([andOperation(parts[...,i],damage[...,0]) for i in range(parts.shape[-1])])


def detect(weights, image, config):
	model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)
	model_WEIGHTS_PATH = os.path.join(MODEL_DIR, weights)
	model.load_weights(model_WEIGHTS_PATH, by_name=True)
	r = model.detect([image], verbose=1)[0]
	return r

def array_to_image(arr):
    im = Image.fromarray(arr.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return str(img_base64, 'utf-8')

def image_to_gcp(image):
    with NamedTemporaryFile() as temp:
        # iName = "".join([str(temp.name),".jpg"])
        filename = 'car-damage-{}.jpg'.format(uuid.uuid4())
        cv2.imwrite(filename,image)
        # image.tofile(gcs_image)
        # gcs_image.seek(0)
        blob = bucket.blob(filename)
        blob.upload_from_filename(filename, content_type='image/jpeg')
        return 'https://storage.cloud.google.com/damaged-car-images/{}?authuser=1'.format(filename)


@app.route("/api", methods=['POST'])
def api():
    args = request.files.get('image')
    image = skimage.io.imread(args)

    r_dent = detect('damage20200116T1049/mask_rcnn_damage_0020.h5',image, config_damage)
    r_scratch = detect('damage20200117T1031/mask_rcnn_damage_0020.h5',image, config_damage)
    rois = np.concatenate((r_dent['rois'], r_scratch['rois']))
    masks = np.concatenate((r_dent['masks'], r_scratch['masks']), axis=2)
    class_ids = np.concatenate((r_dent['class_ids'], r_scratch['class_ids']))
    result = display_instances(image, rois, masks, class_ids)

    r_parts = detect('part20200123T0432/mask_rcnn_part_0010.h5', image, config_parts)
    result_part = display_instances(image, r_parts['rois'], r_parts['masks'], r_parts['class_ids'])
    scratch_parts_masks = map_part_to_damage(r_scratch['masks'],r_parts['masks'])
    dent_parts_masks = map_part_to_damage(r_dent['masks'],r_parts['masks'])

    is_dent = [np.sum(part) for part in dent_parts_masks]
    is_scratch = [np.sum(part) for part in scratch_parts_masks]

    resp =[]
    for i in range(r_parts['class_ids'].size):
        if(is_dent[i]>0 or is_scratch[i]>0):
            dict ={}
            dict['position'] = parts[r_parts['class_ids'][i]]
            severity=[]
            if(is_dent[i]>0):
                severity.append('DENT')
            if(is_scratch[i]>0):
                severity.append('SCRATCH')
            dict['severity'] = severity
            resp.append(dict)
    print(resp)
    damage_response = array_to_image(result)
    part_response = array_to_image(result_part)
    damage_image = image_to_gcp(result)
    part_image = image_to_gcp(result_part)
    
    response = flask.make_response({'response': resp, 'damage_image': str(damage_response), 'part_image': str(part_response),
                                    'damage_image_path': damage_image, 'part_image_path': part_image}, 200)
    response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept, Authorization'
    return response


@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept, Authorization'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)