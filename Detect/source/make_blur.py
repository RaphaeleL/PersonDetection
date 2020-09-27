from PIL import Image, ImageFilter
import glob

images = glob.glob("/Users/raphaele/Desktop/Raf_KI/PANDA-Image/original_cropped_train/image_train/*.jpg")
count = 1
for image in images: 
    print(count)
    img = Image.open(image)
    for i in range(2):
        img = img.filter(ImageFilter.BLUR)
        image = image.replace("image_train", "image_blur")
    img.save(image)
    count += 1