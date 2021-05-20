FROM uhub.service.ucloud.cn/eagle_nest/ubuntu16.04-cuda10.0-cudnn7.4-opencv4.1-snpe1.47-pytorch1.5

RUN rm -rf /usr/local/ev_sdk && mkdir -p /usr/local/ev_sdk
COPY ./ /usr/local/ev_sdk

RUN \
    cd /usr/local/ev_sdk && mkdir -p build && rm -rf build/* \
    && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4 install && rm -rf ../build/*


ENV AUTO_TEST_USE_JI_PYTHON_API=1
RUN python3.6 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /usr/local/ev_sdk/requirements.txt
RUN python3.6 -m pip install /usr/local/ev_sdk/scikit_learn-0.24.1-cp36-cp36m-manylinux2010_x86_64.whl
RUN python3.6 -m pip install /usr/local/ev_sdk/torchvision-0.6.1-cp36-cp36m-manylinux1_x86_64.whl

RUN cd /usr/local/ev_sdk/ && pwd && ls
