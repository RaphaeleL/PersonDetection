import argparse
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='data\images.txt' , help='name of the file in which the image paths are going to be listed, will be created if it does not exist')  
    parser.add_argument('--images_path', type=str, default='data\images\image_train\*.jpg' , help='path of all the images that need to be listed')
    
    opt = parser.parse_args()
    
    with open(opt.file_name, "w+") as axlspd:
        for f in glob.glob(opt.images_path):
            fpath, fname = os.path.split(f)
            rname, extname = os.path.splitext(fname)
            dirtup = (f, rname)
            axlspd.write(f+'\n')