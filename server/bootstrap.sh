#!/bin/bash
yum install -y gcc libstdc++-devel gcc-c++ fuse fuse-devel curl-devel libxml2-devel mailcap automake openssl-devel git docker
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

    mkdir -p ~/buckets/met-office-rmed-forest
    goofys met-office-rmed-forest ~/buckets/met-office-rmed-forest

    cd
    git clone https://github.com/informatics-lab/forest.git
    docker run \
      -p 80:8080 \
      -v /home/ec2-user/forest:/repo/forest \
      -v /home/ec2-user/buckets:/buckets \
      informaticslab/forest bash -c '. /repo/forest/server/run-ec2.sh /repo/forest /buckets'
EOF
