#!/bin/bash
service docker start
usermod -a -G docker ec2-user
exec sudo -u ec2-user /bin/bash - <<EOF
    cd
    git clone https://github.com/s3fs-fuse/s3fs-fuse
    cd s3fs-fuse
    ./autogen.sh
    ./configure --prefix=/usr --with-openssl
    make
    sudo make install

    cd
    curl -L https://github.com/kahing/goofys/releases/latest/download/goofys > goofys
    chmod +x goofys
    sudo mv goofys /usr/bin/goofys

    sudo mkdir -p /s3/met-office-rmed-forest
    sudo goofys -o allow_other met-office-rmed-forest /s3/met-office-rmed-forest

    sudo mkdir /mnt/efs
    sudo mount -t efs fs-b043e241:/ /mnt/efs

    mkdir ~/database
    aws s3 cp s3://met-office-rmed-forest/forest-informaticslab.db ~/database/
    aws s3 cp s3://met-office-rmed-forest/philippines.db ~/database/

    cd
    # Note: /s3/met-office-rmed-forest needs to be mounted since
    #       mounting one directory level up results in a stale inode
    docker run \
      --name forest-container \
      -d \
      -p 80:8080 \
      -v /home/ec2-user/forest:/repo/forest \
      -v /home/ec2-user/database:/database \
      -v /s3/met-office-rmed-forest:/s3/met-office-rmed-forest \
      -v /mnt/efs:/mnt/efs \
      informaticslab/forest bash -c '. /repo/forest/server/run-ec2.sh /repo/forest /s3'

    # Install crontab
    crontab /home/ec2-user/forest/server/housekeeping/cron.file
EOF
