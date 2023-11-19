import cv2
import matplotlib.pyplot as plt
import numpy as np

def display(im_path):
    dpi = 120
    im_data = plt.imread(im_path)
    # print(im_data.shape)
    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

def main():
    image = "Image/book1.jpg"
    img = cv2.imread(image)
    # display(image)

    # inverted_image = cv2.bitwise_not(img)  
    # cv2.imwrite("Image/inverted.jpg", inverted_image)
    # display("Image/inverted.jpg")

    gray_image = grayscale(img)
    # display(gray_image)
    # cv2.imwrite("Image/gray.jpg", gray_image)
    blur = cv2.GaussianBlur(gray_image, (3,3), 0)

    thresh, im_bw = cv2.threshold(blur, 220, 230, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imwrite("temp/bw_image.jpg", im_bw)

    no_noise = noise_removal(im_bw)

    # eroded_image = thin_font(no_noise)

    # no_borders = remove_borders(no_noise)

    while(True):
   
        cv2.imshow('Frame',no_noise)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

