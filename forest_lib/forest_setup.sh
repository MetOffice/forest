# To be run from a linux command line
# after running this script, you should be ready to run forest787

wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh

bash Anaconda3-5.1.0-Linux-x86_64.sh -b -p $INSTALL_DIRECTORY

export PATH=$INSTALL_DIRECTORY/bin/:$PATH

conda create -n forest_dev python=3.6 --yes

source activate forest_dev

conda install -c conda-forge iris=2.0.0 bokeh=0.12.15 cartopy=0.16.0 --yes





