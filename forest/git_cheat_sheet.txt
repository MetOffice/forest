git status
# Saves changes to the local repo
git commit -a -m 'Made tool tip readable'
# Transmits local changes to the remote
git push

# Launches a gui to view revision history
gitk

# Shows what has recently been committed
git log

# Compares uncommitted changes to the local repo
git diff

# setup aliases so I don't need to type 'git checkout' loads of time!
git alias

## Run forest in this currrent setup
## Now run forest
conda activate forest
python setup.py install
# Setup database
# Remember to remove databse file (by default it appends)
# Visualise WRF sample data ...
# forestdb --database claretandy.db /data/users/hadhy/Projects/HIGHWAY/wrf_sampledata.nc
# Some sample files on a local disk ...
forestdb --database claretandy.db /scratch/hadhy/tafrica/CaseStudies/opfc/20190904T0600Z_africa_prods_3236_0.nc #*africa_prods*.nc

forestdb --database rdt_demo.db /scratch/frrn/highway/unified_model/*.nc

# Add dataset to the yaml file (in a text editor)

# check the database
sqlite3 claretandy.db

# Note vld389 is my linux machine name - this allows other computers on the network to see the page
# --dev means that any time a file in the repo is saved, the bokeh server restarts itself
forest --dev --config claretandy.yaml --database claretandy.db --directory /scratch/hadhy/tafrica/CaseStudies/opfc/ --allow-websocket-origin vld389:5006
# For the TMA data ...
#forest --dev --config claretandy.yaml --database claretandy.db --directory /data/users/hadhy/Projects/HIGHWAY/ --allow-websocket-origin vld389:5006

# For the RDT demo ...
forest --dev --config claretandy.yaml --database rdt_demo.db --directory /scratch/frrn/highway/unified_model/ --allow-websocket-origin vld389:5006

# Now go to https://vld389:5006 in a web browser

# Commit stuff back to the branch
git status

# This commits all changed files to the local branch
git commit -a -m 'Desription here'

# Send the branch commits to git hub
git push