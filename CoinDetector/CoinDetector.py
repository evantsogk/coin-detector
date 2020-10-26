from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import filter
import hough_transform


# reads image with input name (e.g. coins002.tif, Image must be in folder 'examples')
try:
    imgName = input('Enter the name of the image: ')
    image = misc.imread('examples/' + imgName, mode='RGB')
except FileNotFoundError:
    sys.exit("Image not found!")
print("\n")

# converts image to grayscale
gray = filter.grayscale(image)

# applies gaussian filter to the image
gaussian = filter.gaussian_filter(gray, 1, 0.8)

# applies sobel filter to the image
sobel = filter.sobel(gaussian)

# converts image to black and white(ready for HT)
bw = filter.black_and_white(sobel)


# uses hough transform to find the coins
def find_coins():

    # find 2 euro coins
    centers2 = hough_transform.ht_circle(bw, 51)
    num2 = len(centers2)
    for c in range(num2):
        coin = Circle((centers2[c][1], centers2[c][0]), 51, fill=False, edgecolor='red')
        ax.add_patch(coin)
    print("There are", num2, "2-Euro coins in the image")

    # find 1 euro coins
    centers1 = hough_transform.ht_circle(bw, 43.35)
    num1 = len(centers1)
    for c in range(num1):
        coin = Circle((centers1[c][1], centers1[c][0]), 43.35, fill=False, edgecolor='blue')
        ax.add_patch(coin)
    print("There are", num1, "1-Euro coins in the image")

    # find 50 cent coins
    centers50 = hough_transform.ht_circle(bw, 46.92)
    num50 = len(centers50)
    for c in range(num50):
        coin = Circle((centers50[c][1], centers50[c][0]), 46.92, fill=False, edgecolor='green')
        ax.add_patch(coin)
    print("There are", num50, "50-Cent coins in the image")

    # find 10 cent coins
    centers10 = hough_transform.ht_circle(bw, 38.25)
    num10 = len(centers10)
    for c in range(num10):
        coin = Circle((centers10[c][1], centers10[c][0]), 38.25, fill=False, edgecolor='purple')
        ax.add_patch(coin)
    print("There are", num10, "10-Cent coins in the image")


# shows images with filters
fig1 = plt.figure(1)
fig1.suptitle("Snapshots of image after applying some filters")
ax1 = fig1.add_subplot(141)
ax2 = fig1.add_subplot(142)
ax3 = fig1.add_subplot(143)
ax4 = fig1.add_subplot(144)
ax1.set_title("1. Grayscale")
ax2.set_title("2. Gaussian")
ax3.set_title("3. Sobel")
ax4.set_title("4. Binary (HT input)")
ax1.imshow(gray, cmap='gray')
ax2.imshow(gaussian, cmap='gray')
ax3.imshow(sobel, cmap='gray')
ax4.imshow(bw, cmap='gray')

# preparing to show coins
fig2, ax = plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(gray, cmap='gray')

find_coins()

plt.show()

