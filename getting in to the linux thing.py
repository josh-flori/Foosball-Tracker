# https://aiyprojects.withgoogle.com/vision-v1/


# RIGHT NOW TO TRAIN MODEL..... DO
cd /users/josh.flori/documents/josh-flori/tensorflow-for-poets-2
# confirm data labels
ls tf_files/training_data
IMAGE_SIZE=160
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=100 \
  --model_dir=tf_files/models/"${ARCHITECTURE}" \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/training_data




python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=/users/josh.flori/desktop/frame_263.jpg




# made a file executable in linux by chmod
chmod +x bonnet_model_compiler.par 

# to get a new docker instance from folder.... using the version of linux with python preinstalled
docker run \
  --name aaqaa3aa \
  -e HOST_IP=$(ifconfig en1 | awk '/ *inet /{print $2}') \
  -v /Users/josh.flori/desktop/send_to_linux:/src \
  -t -i \
  nitincypher/docker-ubuntu-python-pip /bin/bash


# do this to see the contents of the folder you brought into the container.
ls /src


/src/a/bonnet_model_compiler.par \
    --frozen_graph_path=/src/a/frozen_graph.pb \
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