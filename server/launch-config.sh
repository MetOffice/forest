#!/bin/bash
yum install -y git
git clone https://github.com/informatics-lab/forest.git
./forest/server/bootstrap.sh
