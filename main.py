from threading import Thread
import time
import gi
from gi.repository import GLib
from utils import initialize_gstreamer, create_pipeline, on_new_sample

gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Initialize GStreamer
initialize_gstreamer()

# Create the pipeline
pipeline = create_pipeline()

# Get the appsink and connect the frame processing function
appsink = pipeline.get_by_name("sink")
appsink.connect("new-sample", on_new_sample)

# Run GStreamer main loop in a separate thread
main_loop = GLib.MainLoop()
thread = Thread(target=main_loop.run, daemon=True)
thread.start()

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

# Cleanup
pipeline.set_state(Gst.State.NULL)
main_loop.quit()