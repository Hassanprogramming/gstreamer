import gi
import numpy as np
from gi.repository import Gst, GLib
import tensorflow as tf
from PIL import Image as im
# from tensorflow.keras.models import load_model

gi.require_version("Gst", "1.0")

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
MODEL_PATH = "face_net.pb"
graph = load_pb_model(MODEL_PATH)

# Get input and output tensors
input_tensor = graph.get_tensor_by_name("input:0")  # Adjust this based on your model
output_tensor = graph.get_tensor_by_name("embeddings:0")  # Adjust the output tensor name
phase_train_tensor = graph.get_tensor_by_name("phase_train:0")  # Ensure you fetch `phase_train`


### Embedding Extraction with `phase_train`
def get_embedding(face_pixels):
    """Extract embeddings using the .pb model with phase_train set to False."""
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std  # Normalize image
    face_pixels = np.expand_dims(face_pixels, axis=0)  # Add batch dimension

    with tf.compat.v1.Session(graph=graph) as sess:
        # Set phase_train to False during inference
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
    """Creates and returns a GStreamer pipeline with an autovideosink for debugging."""
    pipeline = Gst.parse_launch(
        "v4l2src ! decodebin ! videoconvert ! tee name=t "
        "t. ! queue ! video/x-raw,format=RGB ! appsink name=sink emit-signals=true max-buffers=1 drop=true "
        "t. ! queue ! videoconvert ! autovideosink"
    )
    return pipeline


def on_new_sample(sink):
    """Callback function to process a new frame."""
    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR
    
    buffer = sample.get_buffer()
    caps = sample.get_caps()
    height = caps.get_structure(0).get_int("height")[1]
    width = caps.get_structure(0).get_int("width")[1]

    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR

    # Convert buffer to numpy array
    frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
    buffer.unmap(map_info)

    print("Frame shape:", frame.shape)

    # Resize frame for FaceNet input
    resized_frame = tf.image.resize(frame, (160, 160)).numpy()

    # Extract embedding
    embedding = get_embedding(resized_frame)
    print("Extracted Embedding:", embedding)

    # Save the frame for debugging
    data = im.fromarray(frame)
    data.save('image.png')

    return Gst.FlowReturn.OK
