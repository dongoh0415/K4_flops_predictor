** model SM(FLOPs) 측정기
- .onnx 파읽 읽어서 노드 별 FLOPs 계산 후 total FLOPs 계산
- CNN/MLP model 간단하게 구현해서 test 진행.
----------------------------------------------------------------------------------
** onnx_flops_profiler.py: .onnx file의 flops 측정
** export_cnn_onnx.py : simple_cnn_model.py의 .onnx file(simple_cnn.onnx) export.
** export_mlp_onnx.py: simple_mlp_model.py의 .onnx file(simple_mlp.onnx) export.
** simple_cnn/mlp_model.py: test target models.
----------------------------------------------------------------------------------
command

** for exporting onnx file
python3 export_[model]_onnx.py

** for profiling flops of onnx.
python3 onnx_flops_profiler.py [target_onnx_file]
 
