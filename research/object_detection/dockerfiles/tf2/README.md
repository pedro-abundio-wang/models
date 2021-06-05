# TensorFlow Object Detection on Docker

These instructions are experimental.

## Building and running:

```bash
# From the root of the git repository
sudo docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
sudo docker run --rm --publish-all -it od
```
