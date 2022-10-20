#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) # e.g. 50
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --learning-rate) # e.g. 0.01
      LEARNING_RATE="$2"
      shift # past argument
      shift # past value
      ;;
    --data-directory) # e.g. 0.2
      DATA_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    --out-directory) # e.g. (96,96,3)
      OUT_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$EPOCHS" ]; then
    echo "Missing --epochs"
    exit 1
fi
if [ -z "$LEARNING_RATE" ]; then
    echo "Missing --learning-rate"
    exit 1
fi
if [ -z "$DATA_DIRECTORY" ]; then
    echo "Missing --data-directory"
    exit 1
fi
if [ -z "$OUT_DIRECTORY" ]; then
    echo "Missing --out-directory"
    exit 1
fi

OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

IMAGE_SIZE=$(python3 get_image_size.py --data-directory "$DATA_DIRECTORY")

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into something YOLOX understands)
cd /app/yolox-repo
rm -rf datasets/COCO/
python3 -u /scripts/extract_dataset.py --data-directory $DATA_DIRECTORY --out-directory datasets/COCO/ --epochs $EPOCHS

# train model
python3 -m yolox.tools.train -f datasets/COCO/custom_nano_ti_lite.py -c ../yolox_nano_ti_lite_26p1_41p8_checkpoint.pth -d 0 -b 16 -o -w 1
echo "Training complete"
echo ""

mkdir -p $OUT_DIRECTORY

echo "Converting to ONNX..."
python3 -m yolox.tools.export_onnx -f datasets/COCO/custom_nano_ti_lite.py
# YOLOX has 0..255 inputs, but we want 0..1 (consistent with other models)
# so rewrite the ONNX graph to inject a `Mul` op
python3 /scripts/ei-onnx-tools/inject-mul-255.py --onnx-file ./yolox.onnx --out-file $OUT_DIRECTORY/model.onnx

# also create a second copy with order changed to NHWC (from NCHW) - to be used for the TFLite conversion
cp $OUT_DIRECTORY/model.onnx /tmp/model.onnx
python3 /scripts/ei-onnx-tools/convert-to-nhwc.py --onnx-file /tmp/model.onnx --out-file /tmp/model.onnx
echo "Converting to ONNX OK"
echo ""

# export as f32
echo "Converting to TensorFlow Lite model (fp32)..."
python3 /scripts/ei-onnx-tools/convert-to-tflite.py --onnx-file /tmp/model.onnx --out-file $OUT_DIRECTORY/model.tflite
echo "Converting to TensorFlow Lite model (fp32) OK"
echo ""
