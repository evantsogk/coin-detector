import numpy


# Implementation of Hough Transform that finds circles in an image with radius r
def ht_circle(img, r):

    height, width = img.shape
    r_px = int(round(r))  # radius in actual pixels
    s = 3  # accumulator cell has size 3x3
    votes = numpy.zeros((height, width))  # contains the votes for each circle center

    # +1 because the examined pixel is the center of a 3x3 cell
    # r_px - r_px % s, because number of pixels need to be multiple of s
    # +s to cover the subtraction
    accumulator_radius = 1 + r_px - r_px % s + s
    # number of cells of radius must be a round number
    # so that the examined pixel is at the central cell of the accumulator
    if ((accumulator_radius - 1) // s) % 2 != 0:
        accumulator_radius += s

    # fill the votes
    for y in range(0, height-1):
        for x in range(0, width-1):
            # if the pixel is on an edge
            if img[y, x] == 1:
                # Examine accumulator. The current pixel is the center of the accumulator's central cell
                for a_y in range(y - accumulator_radius, y + accumulator_radius, s):
                    for a_x in range(x - accumulator_radius, x + accumulator_radius, s):
                        # vote cell if the euclidean distance of it's center(a_y,a_x + 1) from current pixel is r ± s/2
                        d = numpy.sqrt((y - (a_y+1))**2 + (x - (a_x+1))**2)
                        if r - s/2 <= d <= r + s/2 and a_y+1 < height and a_x+1 < width:
                            votes[a_y+1, a_x+1] += 1

    # find final circle centers
    max_votes = numpy.max(votes)
    centers = []  # contains the centers of the circles found
    for y in range(0, height):
        for x in range(0, width):
            # a center must have at least 70% of maximum votes to count
            if votes[y, x] > 70/100*max_votes:
                centers.append([y, x])
                # reject centers that have euclidean distance less than 13 pixels
                for c_y in range(y-12, y+12):
                    for c_x in range(x-12, x+12):
                        if votes[c_y, c_x] != 0:
                            d = numpy.sqrt((c_y - y)**2 + (c_x - x)**2)
                            if d < 13:
                                votes[c_y, c_x] = 0
            else:
                votes[y, x] = 0

    return centers

