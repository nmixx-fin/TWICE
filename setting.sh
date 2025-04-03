pip uninstall tensorflow

su -
apt-get install sudo -y
sudo apt-get update


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5

export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio

pip install flash-attn
exit

git clone https://github.com/nmixx-fin/TWICE.git
cd TWICE/
pip install -r requirements.txt