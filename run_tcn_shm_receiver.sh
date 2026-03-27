#!/bin/bash

./holohub run-container --cuda 12 --docker-opts='--ipc=host --shm-size=16gb --mount type=bind,src=/tmp/iceoryx2,dst=/tmp/iceoryx2' tcn_shm_receiver
