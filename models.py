import custom
import custom_multiclass
import os
from mrcnn import model as modellib, utils
import keras.backend as K
import tensorflow as tf

ROOT_DIR = os.getcwd()
config_damage = custom.CustomConfig()
config_parts = custom_multiclass.CustomConfig()
MODEL_DIR = os.path.join(ROOT_DIR, "logs/")
scratch_model_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'damage20200117T1031/mask_rcnn_damage_0020.h5')
dent_model_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'damage20200116T1049/mask_rcnn_damage_0020.h5')
parts_model_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'part20200123T0432/mask_rcnn_part_0010.h5')


class InferenceConfigDamage(config_damage.__class__):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

class InferenceConfigParts(config_parts.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config_damage = InferenceConfigDamage()
config_parts= InferenceConfigParts()

class ScratchModel():
    session = tf.Session()
    K.set_session(session)
    scratch_model = modellib.MaskRCNN(mode="inference", config=config_damage,
                                  model_dir=MODEL_DIR)
    scratch_model.load_weights(scratch_model_WEIGHTS_PATH, by_name=True)
    scratch_model.keras_model._make_predict_function()
    graph = tf.get_default_graph()

class DentModel():
    session = tf.Session()
    K.set_session(session)
    dent_model = modellib.MaskRCNN(mode="inference", config=config_damage,
                                  model_dir=MODEL_DIR)
    dent_model.load_weights(dent_model_WEIGHTS_PATH, by_name=True)
    dent_model.keras_model._make_predict_function()
    graph = tf.get_default_graph()

class PartsModel():
    session = tf.Session()
    K.set_session(session)
    parts_model = modellib.MaskRCNN(mode="inference", config=config_parts,
                                  model_dir=MODEL_DIR)
    parts_model.load_weights(parts_model_WEIGHTS_PATH, by_name=True)
    parts_model.keras_model._make_predict_function()
    graph = tf.get_default_graph()
