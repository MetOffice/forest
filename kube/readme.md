# forest deployment using Kubernetes

This kube/ directory contains the config files for deploying the forest server on AWS infrastructure using the
Kubernetes container manager (https://kubernetes.io/).

The server uses a standard container from the Informatics Lab infrastructure that contains the
required libraries, including python 3.6, iris 2.0, bokeh and cartopy.

The source code for the website and the data to be plotted are both mounted
in the file system of the container.

The source code is the gitgub repository mounted at /github-repos/forest/ using
[this](https://kubernetes.io/docs/concepts/storage/volumes/#gitrepo) capability.

The data, which is stored in an AWS S3 bucket is mounted using a
[goofys](https://github.com/kahing/goofys) [extension written by the informatics lab](https://github.com/informatics-lab/s3-fuse-flex-volume/blob/master/README.md).


## Installing tools
Deploying the server requires some tools to be installed locally. These include
* [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
* [kubernetes-helm](https://github.com/kubernetes/helm)
* [kops](https://github.com/kubernetes/kops)


## Deploying and testing the forest server
Once the relevant kubernetes tools have been installed, there are only a few simple commands to run to depoy the server.
1. Navigate to the kube/ directory which contains this file.
1. run ```helm del --purge forest``` to remove existing server instances.
1. run ```helm install --namespace=forest --name=forest .```

You can now check on the health of the server through the kubernetes cluster dashboard.
``` helm status forest```

Once the server is installed, some changes in the future will only require an upgrade. This can be
executed as follow:
'''helm upgrade forest .```

When making changes, it is useful to ssh into the container to find out what has gone wrong. The
kubectl tool support doing this as follows:
```kubectl -n <NAMESPACE> exec -ti <POD_NAME> <COMAND>```
A specifc example may look like this
```kubectl -n forest exec -ti forest-server-8594c58f7c-mkp6b bash```
where ```forest``` is the namespace, ```forest-server-8594c58f7c-mkp6b``` is the name of the pod
running on the cluster, and ```bash``` is the command to run, in this case a bash shell.