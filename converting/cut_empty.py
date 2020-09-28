import glob, os, argparse

def delete_empty():
    num_delete = 0
    pic_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif', '.webp']
    #Delete train pictures with empty labels
    for f in glob.glob(r'../data/coco/labels/train/**/*.txt', recursive=True):
        fpath, fname = os.path.split(f)
        rname, extname = os.path.splitext(fname)
        dirtup = (f, rname)
        if os.stat(str(f)).st_size == 0 or os.path.getsize(str(f)) == 0:
            for pic_format in pic_formats:
                pic = (str(f).replace('../data/coco/labels/','../data/coco/images/')).replace('.txt',pic_format)
                try:
                    os.remove(pic)
                    os.remove(str(f))
                    num_delete+=1
                    break    
                except:
                    pass
    #Delete train pictures without labels            
    for pic_format in pic_formats:
        for f in glob.glob(r'data/coco/images/train/**/*'+pic_format, recursive=True):
            label = str(f).replace('data/coco/images/','data/coco/labels/').replace(pic_format,".txt")
            try: 
              cache = os.path.getsize(label) 
            except:
              os.remove(f);
              num_delete+=1
        
    print("Finished! Found: ", num_delete, " empty labels")


if __name__ == '__main__':
    print("Start delete pictures with empty labels")
    delete_empty()
