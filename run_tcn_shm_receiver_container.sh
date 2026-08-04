#!/bin/bash

#./holohub run-container --cuda 12 --docker-opts='--ipc=host --shm-size=16gb --mount type=bind,src=/tmp/iceoryx2,dst=/tmp/iceoryx2' tcn_shm_receiver

./holohub run-container --cuda 12 --docker-opts='--ipc=host --shm-size=16gb --mount type=bind,src=/tmp/iceoryx2,dst=/tmp/iceoryx2 --mount type=bind,src=/data/models,dst=/srv/models --mount type=bind,src=/tmp/tcn,dst=/srv/tmp -e HOLOSCAN_ENABLE_PROFILE=1' --add-volume /home/ecku/develop/holoscan/zenoh_config tcn_shm_receiver
