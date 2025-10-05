from rknn.api import RKNN

ONNX_MODEL = './models/yolov8m.onnx'
RKNN_MODEL = './models/yolov8m.rknn'
IMG_PATH = './bus.jpg'
DATASET = './dataset.txt'

QUANTIZE_ON = True

rknn = RKNN(verbose=True)

# pre-process config
print('--> Config model')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
print('done')

# Load ONNX model
print('--> Loading model')
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# Export RKNN model
print('--> Export rknn model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')
rknn.release()