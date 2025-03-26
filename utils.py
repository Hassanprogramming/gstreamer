import gi
import numpy as np
from gi.repository import Gst, GLib
from PIL import Image as im

gi.require_version("Gst", "1.0")

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
        "t. ! queue ! video/x-raw,format=BGR ! appsink name=sink emit-signals=true max-buffers=1 drop=true "
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

    frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
    buffer.unmap(map_info)
    print("this is the frame shape == ", frame.shape)
    
    # creating image object of above np array image
    data = im.fromarray(frame) 
    # saving the final output as a PNG file 
    data.save('image.png') 
    
    return Gst.FlowReturn.OK
