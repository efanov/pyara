run:
	docker build -t cnn_asv_600:v1
    docker run -d -v E:/Audio/ASV/equal_audio:/app/data --name equal equal:test
stop:

container_start:
#docker run -it —rm —gpus all -v $(pwd)/workspace/ image_name:tag bash
#docker run -it --gpus all -v $(pwd):/app/data/ millcool/lstm:v1
docker run -it --gpus all -v /home/ubuntu/model_weights/:/app/results/result -v /home/ubuntu/clips/clips/:/app/data/ -d millcool/full_lstm_l1_40ep:v2

