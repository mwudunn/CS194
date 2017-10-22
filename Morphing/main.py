import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
import scipy
import skimage.io as skio
import pickle
from scipy import interpolate
from scipy.spatial import Delaunay
from skimage.draw import polygon
from scipy.interpolate import RectBivariateSpline



num_points = 24
# read in the image
def read_in(imname):
    return plt.imread(imname)

def select_points(imname, num_points=20):
    location = "./images/"
    filename = imname + "_points" + ".txt"

    # im = plt.imread(location + imname + ".jpg")/255.0

    print ("Pick" + str(num_points) + " points:")
    plt.imshow(im)
    chosen_points = plt.ginput(num_points, timeout=-1)
    plt.close()

    #Corner points
    corner_points = np.array([[0, 0], [0, im.shape[0]-1], [im.shape[1]-1, im.shape[0]-1], [im.shape[1]-1, 0]])
    chosen_points = np.vstack((corner_points, chosen_points))
    

    # print(chosen_points)
    pickle.dump(chosen_points, open("./points/" + filename,"wb"))
    return np.array(chosen_points)

def read_me_points():
    file = "./points/_points" + ".txt"
    if os.path.isfile(file):
        result = np.array(pickle.load(open(file, "rb")))  
        return result
    else:
        return np.array([])

def read_points(imname):
    file = "./points/" + imname + "_points" + ".txt"
    if os.path.isfile(file):
        result = np.array(pickle.load(open(file, "rb")))  
        return result
    else:
        return np.array([])

def read_points_avg(imname, image):
    file = "./average_pts/" + imname + "_points" + ".txt"
    if os.path.isfile(file):
        result = np.array(pickle.load(open(file, "rb")))  
        corner_points = np.array([[0, 0], [0, image.shape[0]-1], [image.shape[1]-1, image.shape[0]-1], [image.shape[1]-1, 0]])
        result = np.vstack((corner_points, result))
        return result
    else:
        return np.array([])

def computeAffine(image_tri, morph_tri):
    image_matrix = np.zeros((3,3))
    morph_matrix = np.zeros((3,3))
    for i in range(3):
        image_matrix[:,i] = np.array([image_tri[i,0], image_tri[i,1], 1])
        morph_matrix[:,i] = np.array([morph_tri[i,0], morph_tri[i,1], 1])
    morph_matrix_inv = np.linalg.inv(morph_matrix)
    return np.dot(image_matrix, morph_matrix_inv)


def get_vertices(points, simplex):
    return np.array([points[i] for i in simplex])


def triangle_mask(height, width, vertices):
    image_triangle = np.zeros((height,width))
    rows = np.array([verts[1] for verts in vertices])
    # print(im_vertices_src)
    cols = np.array([verts[0] for verts in vertices])
    # print(rows)
    # print(cols)
    rr, cc = polygon(rows, cols)
    image_triangle[rr, cc] = 1
    return image_triangle

def midwayWarp(image_src, image_trgt, points_src, points_trgt, avg, triangulation):
    result_image = np.zeros((image_src.shape[0], image_src.shape[1]), dtype=(float, 3))

    for each in triangulation.simplices:
        im_vertices_src = get_vertices(points_src, each)
        im_vertices_trgt = get_vertices(points_trgt, each)
        avg_vertices = get_vertices(avg, each)


        transform_matrix_src = computeAffine(im_vertices_src, avg_vertices)
        transform_matrix_trgt = computeAffine(im_vertices_trgt, avg_vertices)

        image_triangle = triangle_mask(image_src.shape[0], image_src.shape[1], avg_vertices)
        # image_triangle_trgt = triangle_mask(image_trgt.shape[0], image_trgt.shape[1], im_vertices_trgt)

        # plt.imshow(image_triangle)
        # plt.show()
        for j in range(image_src.shape[0]):
            for i in range(image_src.shape[1]):
                if image_triangle[j, i] == 1:
                    new_index_src = np.floor(np.dot(transform_matrix_src, [i, j, 1]))
                    new_index_src = bound(image_src, new_index_src)
                    result_image[j,i] +=  0.5 * image_src[int(new_index_src[1]), int(new_index_src[0])]
                    new_index_trgt = np.floor(np.dot(transform_matrix_trgt, [i, j, 1]))
                    new_index_trgt = bound(image_trgt, new_index_trgt)
                    result_image[j,i] += 0.5 * image_trgt[int(new_index_trgt[1]), int(new_index_trgt[0])]
    return result_image

def morph(image_src, image_trgt, points_src, points_trgt, triangulation, warp_frac, dissolve_frac):
    result_image = np.zeros((image_src.shape[0], image_src.shape[1]), dtype=(float, 3))
    warp = (1 - warp_frac) * points_src + warp_frac * points_trgt

    for each in triangulation.simplices:
        im_vertices_src = get_vertices(points_src, each)
        im_vertices_trgt = get_vertices(points_trgt, each)
        avg_vertices = get_vertices(warp, each)

        transform_matrix_src = computeAffine(im_vertices_src, avg_vertices)
        transform_matrix_trgt = computeAffine(im_vertices_trgt, avg_vertices)

        image_triangle = triangle_mask(image_src.shape[0], image_src.shape[1], avg_vertices)
        # image_triangle_trgt = triangle_mask(image_trgt.shape[0], image_trgt.shape[1], im_vertices_trgt)


        for j in range(image_src.shape[0]):
            for i in range(image_src.shape[1]):
                if image_triangle[j, i] == 1:
                    new_index_src = np.floor(np.dot(transform_matrix_src, [i, j, 1]))
                    new_index_src = bound(image_src, new_index_src)
                    result_image[j,i] += (1 - dissolve_frac) * image_src[int(new_index_src[1]), int(new_index_src[0])]
                    new_index_trgt = np.floor(np.dot(transform_matrix_trgt, [i, j, 1]))
                    new_index_trgt = bound(image_trgt, new_index_trgt)
                    result_image[j,i] += dissolve_frac * image_trgt[int(new_index_trgt[1]), int(new_index_trgt[0])]

    return result_image

def bound(image, arr):
    height = image.shape[0]
    width = image.shape[1]
    if arr[0] >= width:
        arr[0] = width - 1
    elif arr[0] < 0:
        arr[0] = 0
    if arr[1] >= height:
        arr[1] = height - 1
    elif arr[1] < 0:
        arr[1] = 0
    return arr

def main_normal():
    location = "./images/"
    imname1, imname2 = location + sys.argv[1], location + sys.argv[2]
    image1 = read_in(imname1)/255.0
    image2 = read_in(imname2)/255.0

    stripped_name1 = imname1.split(".")[0]
    stripped_name2 = imname2.split(".")[0]
    points1 = read_points(stripped_name1)
    points2 = read_points(stripped_name2)
    if len(points1) != num_points + 4:
        points1 = select_points(imname1, num_points)
    if len(points2) != num_points + 4:
        points2 = select_points(imname2, num_points)
    avg = (points1 + points2) / 2

    triangulation = Delaunay(avg)

    frames = 45
    warp_frac = 0
    dissolve_frac = 0
    increment = 1 / (frames - 1)

    for frame in range(0,frames):
        morph_frame = morph(image1, image2, points1, points2, triangulation, warp_frac, dissolve_frac)
        warp_frac += increment
        dissolve_frac += increment
        plt.imsave("./frames5/morph" + str(frame) + ".jpg", morph_frame)

    # midway_face = midwayWarp(image1, image2, points1, points2, avg, triangulation)
    # plt.imsave("midway_face" + stripped_name1 + stripped_name2 + ".jpg", midway_face)


def morph_average(image_src, points_src, avg, triangulation, warp_frac=1, dissolve_frac=0):
    result_image = np.zeros((image_src.shape[0], image_src.shape[1]), dtype=(float, 3))
    warp = (1 - warp_frac) * points_src + warp_frac * avg

    for each in triangulation.simplices:
        im_vertices_src = get_vertices(points_src, each)
        avg_vertices = get_vertices(warp, each)

        transform_matrix_src = computeAffine(im_vertices_src, avg_vertices)
        image_triangle = triangle_mask(image_src.shape[0], image_src.shape[1], avg_vertices)

        for j in range(image_src.shape[0]):
            for i in range(image_src.shape[1]):
                if image_triangle[j, i] == 1:
                    new_index_src = np.floor(np.dot(transform_matrix_src, [i, j, 1]))
                    new_index_src = bound(image_src, new_index_src)
                    result_image[j,i] += (1 - dissolve_frac) * image_src[int(new_index_src[1]), int(new_index_src[0])]
    return result_image

def main_average():
    location = "./average_images/"
    images = []
    im_points = []
    num_images = 100
    width, height = 0, 0
    for i in range(1,num_images + 1):
        imname = location + str(i) + "b.jpg"
        impts_name = str(i) + "b"
        image = read_in(imname)/255.0
        points = read_points_avg(impts_name, image)
        images.append(image)
        im_points.append(points)
    width = images[0].shape[1]
    height = images[0].shape[0]
    avg = 0
    for each in im_points:
        avg += each
    avg = avg / num_images
    triangulation = Delaunay(avg)

    image = np.zeros((height, width), dtype=(float, 3))
    for i in range(num_images):
        next_image = morph_average(images[i], im_points[i], avg, triangulation)
        image += next_image
        # plt.imsave("./results/warped" + str(i + 1) + ".jpg", next_image)
    image = image / num_images
    plt.imsave("./results/morph_average" + ".jpg", image)


def get_caricature_points():
    location = "./average_images/"
    images = []
    im_points = []
    num_images = 100
    width, height = 0, 0
    for i in range(1,num_images + 1):
        imname = location + str(i) + "b.jpg"
        impts_name = str(i) + "b"
        image = read_in(imname)/255.0
        points = read_points_avg(impts_name, image)
        images.append(image)
        im_points.append(points)
    width = images[0].shape[1]
    height = images[0].shape[0]
    avg = 0
    for each in im_points:
        avg += each
    avg = avg / num_images

    morph_filename = "./results/morph_average.jpg"
    image = read_in(morph_filename)/255.0
    plt.imshow(image)
    for i in range(avg.shape[0]):
        plt.annotate(str(i), xy=avg[i], textcoords='data', size=8)
    plt.show()

def caricature_points(image_me, points, avg, amount=0.2):
    for i in range(len(points)):
        vector_x = points[i][0] - avg[i][0]
        vector_y = points[i][1] - avg[i][1]
        points[i][0] = points[i][0] + vector_x * amount
        points[i][1] = points[i][1] + vector_y * amount
        points[i] = bound(image_me, points[i])
    return points

def main_caricature():
    num_points = 46
    location = "./average_images/"
    images = []
    im_points = []
    num_images = 100
    width, height = 0, 0
    for i in range(1,num_images + 1):
        imname = location + str(i) + "b.jpg"
        impts_name = str(i) + "b"
        image = read_in(imname)/255.0
        points = read_points_avg(impts_name, image)
        images.append(image)
        im_points.append(points)
    width = images[0].shape[1]
    height = images[0].shape[0]
    avg = 0
    for each in im_points:
        avg += each
    morph_avg = avg / num_images

    imname_me = "./results/face"
    image_me = read_in(imname_me + ".jpg")/255.0

    points = read_me_points()
    if len(points) != num_points + 4:
        points = select_points(imname_me + ".jpg", num_points)
    warp_amount = 0.4
    warp_points = caricature_points(image_me, np.copy(points), morph_avg, warp_amount)
    triangulation = Delaunay(warp_points)
    result_image = morph_average(image_me, points, warp_points, triangulation)
    result_image.astype("uint8")


    plt.imsave("./results/caricature" + str(warp_amount) + ".jpg", result_image)

def main_genderswap():
    location = "./images/"
    imname1, imname2 = location + "face_test.jpg", location + "random_face_test.jpg"
    image1 = read_in(imname1)/255.0
    image2 = read_in(imname2)/255.0

    stripped_name1 = "face_test"
    stripped_name2 = "random_face_test"

    points1 = read_points(stripped_name1)
    points2 = read_points(stripped_name2)
    if len(points1) != num_points + 4:
        points1 = select_points("face_test", num_points)
    if len(points2) != num_points + 4:
        points2 = select_points("random_face_test", num_points)
    triangulation = Delaunay(points2)

    result_image = morph_average(image1, points1, points2, triangulation)
    plt.imsave("./results/genderswap.jpg", result_image)

# main_normal()
# main_average()
# main_caricature()
# main_genderswap()