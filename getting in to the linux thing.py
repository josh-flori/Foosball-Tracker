# https://aiyprojects.withgoogle.com/vision-v1/
# made a file executable in linux by chmod
chmod +x bonnet_model_compiler.par 

# to get a new docker instance from folder.... using the version of linux with python preinstalled
docker run \
  --name aaqaaaaaaaa \
  -e HOST_IP=$(ifconfig en1 | awk '/ *inet /{print $2}') \
  -v /Users/josh.flori/desktop/send_to_linux:/src \
  -t -i \
  nitincypher/docker-ubuntu-python-pip /bin/bash


# do this to see the contents of the folder you brought into the container.
ls /src


/src/a/bonnet_model_compiler.par \
    --frozen_graph_path=/src/a/mobilenet_v1_160res_0.5_imagenet.pb \
    --output_graph_path=/cat_detector.binaryproto \
    --input_tensor_name='input' \
    --output_tensor_names='MobilenetV1/Predictions/Softmax' \
    --input_tensor_size=160 \
    --debug


/src/a/bonnet_model_compiler.par \
    --frozen_graph_path=/src/a/mobilenet_v1_160res_0.5_imagenet.pb \
    --output_graph_path=./mobilenet_v1_160res_0.5_imagenet.binaryproto \
    --input_tensor_name="input" \
    --output_tensor_names="MobilenetV1/Predictions/Softmax" \
    --input_tensor_size=160


IMAGE_SIZE=160

  
  
  
  
./bonnet_model_compiler.par  --frozen_graph_path=./saved_model.pb --output_graph_path=./rounded_graph.binaryproto  --input_tensor_name="input" --output_tensor_names="final_result" --input_tensor_size=150528 --debug  



tar -zxvf tf_models_2018_05_15.tgz