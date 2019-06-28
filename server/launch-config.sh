#!/bin/bash
yum install git docker -y
service docker start
usermod -a -G docker ec2-user
mkdir /s3
exec sudo -u ec2-user /bin/bash - <<EOF
    cd
    git clone https://github.com/informatics-lab/forest.git
    docker run \
      -p 80:8080 \
      -v /home/ec2-user/forest:/repo/forest \
      -v /s3:/s3 \
      informaticslab/forest bash -c '. /repo/forest/server/run-ec2.sh /repo/forest'
EOF
