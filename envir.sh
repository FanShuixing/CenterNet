COCOAPI=../
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
make 
python setup.py install --user

cd /output/CenterNet
pip install -r requirements.txt
cd src/lib/models/networks/DCNv2
./make.sh