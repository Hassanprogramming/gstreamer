import gi
import numpy as np
from settings import load_config
from gi.repository import Gst, GLib
import tensorflow as tf
from PIL import Image as im
# from tensorflow.keras.models import load_model

gi.require_version("Gst", "1.0")

config = load_config()

### Load the .pb Model ###
def load_pb_model(pb_model_path):
    """Load a frozen graph (.pb) model into a TensorFlow session."""
    with tf.io.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph

# Path to your .pb model
MODEL_PATH = config["facenet"]["model_path"]
graph = load_pb_model(MODEL_PATH)

# Get input and output tensors
input_tensor = graph.get_tensor_by_name(config["facenet"]["input_tensor_name"])
output_tensor = graph.get_tensor_by_name(config["facenet"]["output_tensor_name"])
phase_train_tensor = graph.get_tensor_by_name(config["facenet"]["phase_train_tensor_name"])


### Embedding Extraction with `phase_train`
def get_embedding(face_pixels):
    """Extract embeddings using GPU acceleration if enabled in config.yaml."""
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std  # Normalize image
    face_pixels = np.expand_dims(face_pixels, axis=0)  # Add batch dimension

    device = "/GPU:0" if config["tensorflow"]["use_gpu"] else "/CPU:0"

    with tf.device(device):  # Run on GPU or CPU based on config
        with tf.compat.v1.Session(graph=graph) as sess:
            embedding = sess.run(
                output_tensor, 
                feed_dict={input_tensor: face_pixels, phase_train_tensor: False}
            )
    
    return embedding[0]



def initialize_gstreamer():
    """Initializes GStreamer."""
    Gst.init()

# def create_pipeline():
#     """Creates and returns a GStreamer pipeline with an appsink."""
#     pipeline = Gst.parse_launch(
#         "v4l2src ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=true max-buffers=1 drop=true"
#     )
#     return pipeline


#### visual the video #### !hint==for debug
def create_pipeline():
    """Creates and returns a GStreamer pipeline based on config."""
    debug_mode = config["gstreamer"]["debug_mode"]
    pipeline_str = (
        f"v4l2src device={config['gstreamer']['device']} ! decodebin ! videoconvert ! tee name=t "
        f"t. ! queue ! video/x-raw,format={config['gstreamer']['frame_format']} ! appsink name={config['gstreamer']['appsink_name']} "
        f"emit-signals=true max-buffers={config['gstreamer']['max_buffers']} drop={str(config['gstreamer']['drop_buffers']).lower()} "
    )
    
    if debug_mode:
        pipeline_str += "t. ! queue ! videoconvert ! autovideosink"
    
    return Gst.parse_launch(pipeline_str)


def on_new_sample(sink):
    """Callback function to process a new frame."""
    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR
    
    buffer = sample.get_buffer()
    caps = sample.get_caps()
    
    _, height = caps.get_structure(0).get_int("height")
    _, width = caps.get_structure(0).get_int("width")

    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR

    frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
    buffer.unmap(map_info)

    resized_frame = tf.image.resize(frame, (config["facenet"]["image_size"], config["facenet"]["image_size"])).numpy()

    embedding = get_embedding(resized_frame)
    print("Extracted Embedding:", embedding)

    if config["image_processing"]["save_debug_image"]:
        data = im.fromarray(frame)
        data.save(config["image_processing"]["debug_image_path"])

    return Gst.FlowReturn.OK

