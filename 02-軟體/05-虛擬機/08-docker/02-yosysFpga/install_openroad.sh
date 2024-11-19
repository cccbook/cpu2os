# https://github.com/The-OpenROAD-Project/OpenROAD/blob/master/docs/user/Build.md

git clone --recursive https://github.com/The-OpenROAD-Project/OpenROAD.git
cd OpenROAD

sudo ./etc/DependencyInstaller.sh

mkdir build && cd build
cmake ..
make
sudo make install 

