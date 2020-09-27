cd ../config/cfg 
wget "https://www.dropbox.com/s/dfwvrb4nph4w9rk/yolov3-spp.cfg"
wget "https://www.dropbox.com/s/bnnnok572sg5sqz/yolov3-tiny.cfg"
cd ../data 
wget "https://www.dropbox.com/s/djkrmubfgo24ur3/coco.names" 
cd ../weights
wget "https://www.dropbox.com/s/nzmibytp2hmv666/yolov3_tiny_best.pt"
wget "https://www.dropbox.com/s/jdhx0ivslj9zlxd/yolov3_best.pt"
wget "https://www.dropbox.com/s/kb79bx1utukho7s/yolov3_last.pt"
cd ../../data/original_cropped_valid/image_annos
wget "https://www.dropbox.com/s/d3mak1889a7jby8/valid_coco.json"
wget "https://www.dropbox.com/s/ga28wbt6u4verya/valid.json" 
cd ..
wget "https://www.dropbox.com/s/nwehy278dybo2m8/image_valid.zip"
unzip image_valid.zip
rm -rf image_valid.zip
rmm -rf __MACOSX
cd ../..
