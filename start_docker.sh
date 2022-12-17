sudo systemctl start docker
sudo chmod 666 /var/run/docker.sock
docker-compose run --rm vsgan_tensorrt