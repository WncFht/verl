docker rm -f verl 2>/dev/null
docker run -d --rm \
  --runtime=nvidia \
  --gpus all \
  --net=host \
  --shm-size="100g" \
  --cap-add=SYS_ADMIN \
  --user "$(id -u):$(id -g)" \
  --group-add 1005 \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /mnt:/mnt \
  -v /mnt/data2:/mnt/data2 \
  -v /home/fanghaotian:/home/fanghaotian \
  -v /mnt/data2/fanghaotian/src:/home/fanghaotian/src \
  --name verl \
  docker.1ms.run/verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 \
  sleep infinity