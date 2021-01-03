# tensorflow-frozen-graph-to-tensorrt-graph
This repo is all about how to convert frozen TF graph to Tensorrt. If we have a frozen TF graph then we can optimize it before using it for inferences. To optimize we can convert frozen TF graph into tensorrt frozen graph. We can use INT8, FP16, FP32, etc. precision mode.

To convert TF to tensorrt run the following command in terminal: <br>
```python tf_frozen_to_tfrt.py```

make sure you have changed the frozen model directory/path, output node name, and path/directory where you want to save the final tensorrt graph.
