model_transform.py \
  --model_name bge_large_512 \
  --model_def text2vec-bge-large-chinese.onnx \
  --input_shapes [[4,512],[4,512],[4,512]] \
  --mlir bge_large_512.mlir

  model_deploy.py \
  --mlir bge_large_512.mlir \
  --quantize F16 \
  --chip bm1684x \
  --model bge_large_512_fp16.bmodel \
  --compare_all \
  --debug
