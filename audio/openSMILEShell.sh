sudo tar -zxvf opensmile-2.3.0.tar.gz
sudo sed -i '117s/(char)/(unsigned char)/g' opensmile-2.3.0/src/include/core/vectorTransform.hpp
sudo apt-get update
sudo apt-get install autoconf automake libtool m4 gcc
sudo cd opensmile-2.3.0
sudo bash buildStandalone.sh
