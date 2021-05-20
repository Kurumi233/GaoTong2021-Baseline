python3.6 /usr/local/ev_sdk/convert_onnx.py
pwd
ls /usr/local/ev_sdk/model

mkdir /usr/local/ev_sdk/pic
mkdir /usr/local/ev_sdk/raw
wget -O /usr/local/ev_sdk/pic/0.jpg https://www.baidu.com/img/bd_logo1.png

python /usr/local/ev_sdk/create_onnx_raw.py -i /usr/local/ev_sdk/pic/ -d /usr/local/ev_sdk/raw -s 224
python /usr/local/ev_sdk/tmp.py # 生成list的

source /opt/snpe/snpe_venv/bin/activate

pip install onnx
pip install protobuf --upgrade
pip install opencv-python

snpe-onnx-to-dlc\
    --input_network /usr/local/ev_sdk/model/model.onnx\
    --output_path /usr/local/ev_sdk/model/model.dlc\
    -d "input" "1,3,224,224"\
    --out_node "output"\
&& echo "snpe-onnx-to-dlc done."

snpe-dlc-quantize\
    --input_dlc /usr/local/ev_sdk/model/model.dlc\
    --input_list /usr/local/ev_sdk/file_list.txt\
    --output_dlc /usr/local/ev_sdk/model/model_quantized.dlc\
    --input_dlc --enable_htp\
&& echo "snpe-dlc-quantize done."

rm /usr/local/ev_sdk/model/model.dlc