cd ..
cd cfg 
rm yolov3-spp.cfg
rm yolov3-tiny.cfg
cd ../data 
rm coco.names
cd ../weights
rm yolov3_tiny_best.pt
rm yolov3_best.pt
rm yolov3_last.pt
cd ../original_cropped_valid/image_annos
rm valid_coco.json
rm valid.json
cd ..
rm -rf __MACOSX/
rm -rf image_valid
rm -rf image_valid.zip
cd ..