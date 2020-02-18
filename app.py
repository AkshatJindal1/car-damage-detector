# API Specific
from flask import Flask, request, make_response
import flask

#Model Specific
import numpy as np
import cv2
import os
import skimage
from mrcnn import visualize
from PIL import Image
import io
import base64
from models import ScratchModel, DentModel, PartsModel
import keras.backend as K
import tensorflow as tf
import time

#GCP specific
import uuid
from google.cloud import storage
from tempfile import NamedTemporaryFile

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'

client = storage.Client().create_anonymous_client()
bucket = client.get_bucket('damaged-car-images')


app = Flask(__name__)
ROOT_DIR = os.getcwd()

parts = {
    1: "rear_bumper",
    2: "front_bumper",
    3: "headlamp",
    4: "door",
    5: "hood"
}

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

def detect(model_type, image):
    if(model_type=='scratch'):
        K.set_session(ScratchModel.session)
        with ScratchModel.session.as_default():
            with ScratchModel.graph.as_default():
                ScratchModel.session.run(tf.global_variables_initializer())
                r=ScratchModel.scratch_model.detect([image], verbose=1)[0]
        return r
    elif(model_type == 'dent'):
        K.set_session(DentModel.session)
        with DentModel.session.as_default():
            with DentModel.graph.as_default():
                DentModel.session.run(tf.global_variables_initializer())
                r=DentModel.dent_model.detect([image], verbose=1)[0]
        return r
    else:
        K.set_session(PartsModel.session)
        with PartsModel.session.as_default():
            with PartsModel.graph.as_default():
                PartsModel.session.run(tf.global_variables_initializer())
                r=PartsModel.parts_model.detect([image], verbose=1)[0]
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
    start_time = time.time()
    args = request.files.get('image')
    image = skimage.io.imread(args)

    r_dent = detect('dent', image)
    r_scratch =  detect('scratch', image)
    rois = np.concatenate((r_dent['rois'], r_scratch['rois']))
    masks = np.concatenate((r_dent['masks'], r_scratch['masks']), axis=2)
    class_ids = np.concatenate((r_dent['class_ids'], r_scratch['class_ids']))
    result = display_instances(image, rois, masks, class_ids)

    r_parts = detect('parts', image)
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

    damage_response = array_to_image(result)
    part_response = array_to_image(result_part)
    damage_image= 'abcd'
    part_image = 'abcd'
    damage_image = image_to_gcp(result)
    part_image = image_to_gcp(result_part)
    total_time ="--- %s seconds ---"% (time.time()- start_time)
    response = flask.make_response({'response': resp, 'damage_image': str(damage_response), 'part_image': str(part_response),
                                    'damage_image_path': damage_image, 'part_image_path': part_image, 'time_taken': total_time}, 200)
    response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept, Authorization'
    return response


@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept, Authorization'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)