import torch.onnx
import numpy

dummy_input = torch.randn(1, 3, 16, 128, 128, device='cuda')
model = torch.load("F:/中国手语数据集科大/SLR/cnn3d_models/slr_cnn3d_epoch003.pth")

# Export the model   
torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "slr_cnn3d_epoch005.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        # opset_version=10,    # the ONNX version to export the model to 
        # do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
        # dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
        #                     'modelOutput' : {0 : 'batch_size'}}''
        verbose=True
        )
