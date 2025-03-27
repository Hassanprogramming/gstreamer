# Video Streaming with GStreamer and Python

This project demonstrates how to use GStreamer, GLib, and Python's threading module to stream video from a webcam or an IP camera. The script sets up a GStreamer pipeline to capture video, decode it, convert it to the appropriate format, process frames for visualization (e.g., adding bounding boxes or lines), and display it in an external window.

## Requirements

- Python 3.x
- GStreamer (1.0 or later)
- Python bindings for GStreamer (PyGObject)
- Pillow (for image processing)
- NumPy (for frame manipulation)
- TensorFlow (for running the FaceNet model)

### Installing Dependencies

Before running the script, ensure that the following dependencies are installed:

1. **Install GStreamer:**

   For Linux (Ubuntu/Debian):
   ```bash
   sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
   ```
   
2. **Install Python dependencies:**
   ```bash
   pip install PyGObject numpy pillow
   ```

## Features

- Captures video using GStreamer from a webcam or an IP camera.
- Uses an `appsink` element to process frames in Python.
- Displays real-time video using `autovideosink` for debugging.
- Saves frames as PNG images for further analysis.
- Supports frame processing, including modifying images using Pillow (PIL).

## Usage

Run the Python script to start the video streaming:
```bash
python3 main.py
```

### Debug Mode (Frame Visualization)

To enable debugging and see processed frames:
- The GStreamer pipeline uses a `tee` element to split the video stream, sending one to an `appsink` for processing and the other to `autovideosink` for real-time viewing.
- Frames are saved as `image.png` for further analysis.

Example snippet inside `on_new_sample`:
```python
from PIL import Image as im

def on_new_sample(sink):
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
    print("this is the frame shape ==", frame.shape)
    
    # Save the frame as an image
    data = im.fromarray(frame)
    data.save('image.png')
    
    return Gst.FlowReturn.OK
```

## Troubleshooting

### GStreamer Version Warning
If you see a warning like:
```
PyGIWarning: Gst was imported without specifying a version first.
```
Make sure `gi.require_version('Gst', '1.0')` is called before importing `Gst`.

### OpenGL Errors (GTK/Gdk)
If you see errors related to OpenGL (`gdk_gl_context_make_current() failed`), try setting:
```bash
export GDK_BACKEND=x11
```
before running the script.

## Future Enhancements

- Integrate object detection models for real-time processing.
- Improve visualization with interactive UI elements.
- Optimize performance for high-resolution video streams.

---
This project is designed for real-time video streaming and processing using Python and GStreamer. Contributions and improvements are welcome!

