# TI YOLOX transfer learning model for Edge Impulse

This repository is an example on how to [bring your own model](https://docs.edgeimpulse.com/docs/adding-custom-transfer-learning-models) into Edge Impulse. This repository is using TI's fork of YOLOX (an object detection model), but the same principles apply to other transfer learning models.

As a primer, read the [Bring your own model](https://docs.edgeimpulse.com/docs/adding-custom-transfer-learning-models) page in the Edge Impulse docs.

What this repository does (see [run.sh](run.sh)):

1. Convert the training data / training labels into YOLOX format using [extract_dataset.py](extract_dataset.py).
1. Train YOLOX model.
1. Convert the YOLOX model into ONNX & TFLite formats.
1. Done!

## Running the pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16.0 or higher.
3. Create a new Edge Impulse project, and make sure the labeling method is set to 'Bounding boxes'.
4. Add and label some data.
5. Under **Create impulse** set the image size to e.g. 160x160, 320x320 or 640x640, add an 'Image' DSP block and an 'Object Detection' learn block.
6. Open a command prompt or terminal window.
7. Initialize the block:

    ```
    $ edge-impulse-blocks init
    # Answer the questions:
    #   Select "Object Detection" for 'What type of data does this model operate on?'
    #   Select "YOLOX" for 'What's the last layer...'
    ```

    (If YOLOX does not show up, then update the Edge Impulse CLI)

8. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

9. Build the container:

    ```
    $ docker build -t yolox .
    ```

10. Run the container to test the script (you don't need to rebuild the container if you make changes):

    ```
    $ docker run --rm -it -v $PWD:/scripts -v $PWD/yolox-repo:/app/yolox-repo yolox --data-directory data --out-directory out --epochs 30 --learning-rate 0.01
    ```

11. This creates an .onnx and a .tflite file in the 'out' directory.

12. Now you can test inference as well:

    ```
    $ python3 tflite-inference-test.py
    ```

#### Adding extra dependencies

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

## Fetching new data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16 or higher.
2. Open a command prompt or terminal window.
3. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

> **Note: to fetch data from another project** run the following:
> ```
> $ edge-impulse-blocks --clean
> # Then follow the steps provided
> $ edge-impulse-blocks runner --download-data data/
> # Then choose your project
> ```

## Pushing the block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block:

    ```
    $ edge-impulse-blocks push
    ```

2. The block is now available under any of your projects, via  **Create impulse > Add learning block**.
