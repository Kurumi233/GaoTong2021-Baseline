from model import BaseModel
import torch.onnx


if __name__ == '__main__':
    pth = '/usr/local/ev_sdk/model/best.pth'
    
    device = torch.device("cpu")
    
    model = BaseModel(model_name='rep-a2')
    model.load_state_dict(torch.load(pth)['net'])
    model.to(device)
    model.eval()

    batch_size = 1  #批处理大小
#     input_shape = (3, 64, 64)
    input_shape = (3, 224, 224)   #输入数据,改成自己的输入shape

    # #set the model to inference mode

    x = torch.randn(batch_size, *input_shape)   # 生成张量
    x = x.to(device)
    export_onnx_file = "/usr/local/ev_sdk/model/model.onnx"  # 目的ONNX文件名
    print('convert onnx model to - {}'.format(export_onnx_file))
    torch.onnx.export(model,
                        x,
                        export_onnx_file,
                        opset_version=8,
#                         verbose=True,
                        do_constant_folding=False, # 是否执行常量折叠优化
                        input_names=["input"], # 输入名
                        output_names=["output"], # 输出名
                        )
    
    
    