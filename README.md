# Video Streaming with GStreamer and Python

This project demonstrates how to use GStreamer, GLib, and Python's threading module to stream video from a webcam or ip_camera. The script sets up a GStreamer pipeline to capture video from a webcam, decode it, convert it to the appropriate format, and display it in an external window.

## Requirements

- Python 3.x
- GStreamer (1.0 or later)
- Python bindings for GStreamer (PyGObject)

### Installing Dependencies

Before running the script, ensure that the following dependencies are installed:

1. **Install GStreamer:**

   For Linux (Ubuntu/Debian):
   ```bash
   sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
