from PIL import Image, ImageFilter
import glob

images = glob.glob("../data/images/valid/*.jpg")
count = 1
for image in images: 
    print(count)
    img = Image.open(image)
    for i in range(2):
        img = img.filter(ImageFilter.BLUR)
        image = image.replace("image_train", "image_blur")
    img.save(image)
    count += 1