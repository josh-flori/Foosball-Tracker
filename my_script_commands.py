# my script commands


scp pi@raspberrypi.local:~/test.h264 /users/josh.flori/desktop
scp pi@raspberrypi.local:~/image.jpg /users/josh.flori/desktop
scp pi@raspberrypi.local:~/mymodel.binaryproto /users/josh.flori/desktop


/users/josh.flori/desktop/test/bin/python3 /users/josh.flori/documents/josh-flori/foosball-tracker/vid_to_frames.py  -v='/users/josh.flori/desktop/test.h264' -o='/users/josh.flori/desktop/training_data/'
/users/josh.flori/desktop/test/bin/python3 /users/josh.flori/documents/josh-flori/foosball-tracker/normalize.py  -i='/users/josh.flori/desktop/training_data/'

cd /users/josh.flori/documents/josh-flori/tensorflow-for-poets-2

IMAGE_SIZE=160
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/"${ARCHITECTURE}" \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=/users/josh.flori/desktop/training_data/

# for testing after trained
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/aa.jpg \
    --input_height=160 \
    --input_width=160
  
  
  
docker run \
  --name baa \
  -e HOST_IP=$(ifconfig en1 | awk '/ *inet /{print $2}') \
  -v /Users/josh.flori/desktop/send_to_linux/:/src \
  -t -i \
  nitincypher/docker-ubuntu-python-pip /bin/bash  
  
  
/src/bonnet_model_compiler.par \
    --frozen_graph_path=/src/frozen_graph.pb \
    --output_graph_path=/mymodel.binaryproto \
    --input_tensor_name='input' \
    --output_tensor_names='MobilenetV1/Predictions/Softmax' \
    --input_tensor_size=160 \
    --debug  
    
docker cp a759aad27e7b:/mymodel.binaryproto /users/josh.flori/desktop/


scp /users/josh.flori/desktop/mymodel.binaryproto /users/josh.flori/desktop/send_to_linux/retrained_labels.txt pi@raspberrypi.local:~/


~/AIY-projects-python/src/examples/vision/mobilenet_based_classifier.py \
  --model_path ~/mymodel.binaryproto \
  --label_path ~/retrained_labels.txt \
  --input_height 160 \
  --input_width 160 \
  --top_k 2 \
  --input_layer input \
  --output_layer final_result \
  --threshold 0 \
  --preview
  
  
  
~/AIY-projects-python/src/examples/vision/image_classification.py
  
  
nano AIY-projects-python/src/examples/vision/mobilenet_based_classifier.py





# 1640, 1232









python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=/users/josh.flori/desktop/aa.jpg \
    --input_height=160 \
    --input_width=160
    
    
    
    
    
    
    
    
    
aiyvision.inference 
inference.CameraInference(model...) inputnormalizer
nano ~/AIY-projects-python/src/aiy/vision/inference.py




(a, args.input_std)








I am trying to deploy my own image recognition model on the vision kit and test it using mobilenet_based_classifier.py 
I have followed all of the instructions, created a model.binaryproto with a labels.txt, placed on the pi and have ran the classifier, but the only thing it returns is ___________





    