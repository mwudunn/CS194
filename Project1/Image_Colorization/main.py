# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import sys
from skimage import transform
from skimage import color



# read in the image
def read_in(imname):
	return skio.imread(imname)

# convert to double (might want to do this later on to save memory)
def convert(im):
	return sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
def generate_color_channels(im):
	height = int(np.floor(im.shape[0] / 3.0))
	b = im[:height]
	g = im[height: 2*height]
	r = im[2*height: 3*height]
	return r, g, b

# separate color channels

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

def find_frame(im, x_pos, y_pos, frame_size):
	frame = im[int(y_pos - frame_size//2):int(y_pos + frame_size//2), int(x_pos - frame_size//2):int(x_pos + frame_size//2)]
	return frame

def norm_cc(image1, image2):
	return np.sum(np.multiply(image1 - np.mean(image1), image2 - np.mean(image2))/np.multiply(np.std(image1), np.std(image2)))

def align(channel_2, channel_1, window_size=30, frame_size=100):
	height, width = np.floor(channel_1.shape[0]), np.floor(channel_1.shape[1])
	frame_1 = find_frame(channel_1, width // 2, height // 2, frame_size)
	min_ss = -1
	displacement_coords = None
	for y in range(-window_size, window_size + 1):
		y_pos = int(height // 2 + y)
		for x in range(-window_size, window_size + 1):
			x_pos = int(width // 2 + x)
			frame_2 = find_frame(channel_2, x_pos, y_pos, frame_size)
			new_ss = norm_cc(frame_1, frame_2)
			if new_ss > min_ss or min_ss == -1:
				min_ss = new_ss
				displacement_coords = np.array([-y, -x])
	return displacement_coords

### ag = align(g, b)
### ar = align(r, b)
# create a color image

def roll_image(im, displacement_coords):
	return np.roll(np.roll(im, displacement_coords[0], axis = 0), displacement_coords[1], axis = 1)


def image_pyramid(channel_2, channel_1, window_size=30, frame_size=150):
	if channel_2.shape[0] <= 600 or channel_2.shape[1] <= 600:
		return align(channel_2, channel_1, window_size, frame_size)
	else:
		resized_channel_1 = sk.transform.rescale(channel_1, 0.5)
		resized_channel_2 = sk.transform.rescale(channel_2, 0.5)
		next_displacement_coords = image_pyramid(resized_channel_2, resized_channel_1, window_size, frame_size)
		next_displacement_coords *= 2
		adjusted_channel_2 = roll_image(channel_2, next_displacement_coords)
		aligned = align(adjusted_channel_2, channel_1, window_size, frame_size)
		return np.add(aligned, next_displacement_coords)

def crop(image, iterations=2, crop_sensitivity=1):
	if iterations == 0:
		return image
	#size of scan window should change relative to size of image, but be large enough to detect borders
	scan_window_size = max(int(min(image.shape[0] * 0.06, image.shape[1] * 0.06)), 20)

	#Change image to grayscale
	grey_image = sk.color.rgb2grey(image)
	#Moves across image based on the border we are looking for
	def detect_border(initial_x, initial_y, dir_x, dir_y):
		greatest_luminance_difference = -1
		cumulative_luminance_difference = 0
		border_index = -1
		previous_pixel_slice = None
		if dir_x == 0:
			for y_dis in range(scan_window_size):
				pixel_slice = grey_image[initial_y + (dir_y * y_dis), initial_x - scan_window_size // 2: initial_x + scan_window_size // 2]
				if previous_pixel_slice is not None:
					difference = abs(np.sum(np.array(pixel_slice) - np.array(previous_pixel_slice))) / scan_window_size
					if difference > greatest_luminance_difference:
						greatest_luminance_difference = difference
						border_index = y_dis
					cumulative_luminance_difference += difference
				previous_pixel_slice = pixel_slice
		else:
			for x_dis in range(scan_window_size):
				pixel_slice = grey_image[initial_y - scan_window_size // 2: initial_y + scan_window_size // 2, initial_x + (dir_x * x_dis)]
				if previous_pixel_slice is not None:
					difference = abs(np.sum(pixel_slice - previous_pixel_slice)) / scan_window_size
					if difference > greatest_luminance_difference:
						greatest_luminance_difference = difference
						border_index = x_dis
					cumulative_luminance_difference += difference
				previous_pixel_slice = pixel_slice
		if greatest_luminance_difference < 0.1 and (cumulative_luminance_difference / scan_window_size) < crop_sensitivity / 125:
			return 0
		return border_index


	top_index = detect_border(grey_image.shape[1] // 2, 0, 0, 1)
	bottom_index = detect_border(grey_image.shape[1] // 2, grey_image.shape[0] - 1, 0, -1)
	left_index = detect_border(0, grey_image.shape[0] // 2, 1, 0)
	right_index = detect_border(grey_image.shape[1] - 1, grey_image.shape[0] // 2, -1, 0)


	cropped_image = image[top_index: image.shape[0] - bottom_index, left_index: image.shape[1] - right_index]
	return crop(cropped_image, iterations-1)

#Note: r, g, b refers to the displaced images
# def automatic_white_balance(im, window_size=30):
# 	y, x = im.shape[0] // 2, im.shape[1] // 2

# 	def compute_avg(image, start_y, start_x):
# 		window = im[y - window_size // 2: y + window_size // 2, x - window_size // 2: x + window_size // 2]
# 		average_colors = (np.sum(window, axis=(0,1)))
# 		for i in range(len(average_colors)):
# 			average_colors[i] = average_colors[i] / (window_size * window_size)
# 		return average_colors
# 	average = compute_avg(im, y, x)
# 	grey_image = sk.color.rgb2grey(im)
# 	grey_average = compute_avg(grey_image, y, x)
# 	print(grey_average)
# 	print(average)
# 	scalar = np.sum(average) / np.sqrt(np.sum(np.multiply(average, average)))
# 	print(scalar)
# 	red_mult = scalar / average[0]
# 	green_mult = scalar / average[1]
# 	blue_mult = scalar / average[2]
# 	result = np.multiply(im, [red_mult, green_mult, blue_mult])

	


def generate_color_image(r, g, b):
	if r.shape[0] > 1200 or r.shape[1] > 1200:
		disp_ag = image_pyramid(g, b)
		disp_ar = image_pyramid(r, b)
	else:
		disp_ag = align(g, b)
		disp_ar = align(r, b)
	print(disp_ag)
	print(disp_ar)
	ag = roll_image(g, disp_ag)
	ar = roll_image(r, disp_ar)
	combined = np.dstack([ar, ag, b])
	# automatic_white_balance(combined)
	return combined

# save the image and display
def display(im_out, output_file):
	output_file += ".jpg"
	fname = str(output_file)
	skio.imsave(fname, im_out)
	# display the image
	skio.imshow(im_out)
	skio.show()


def main(imname, b_crop):
	im = read_in(imname)
	im = convert(im)
	r, g, b = generate_color_channels(im)
	im_out = generate_color_image(r, g, b)
	output_file = imname.split('.')[0]
	output_file += "_colorized"
	if b_crop.lower() == "true":
		if r.shape[0] > 1200 or r.shape[1] > 1200:
			im_out = crop(im_out, 7)
		else:
			im_out = crop(im_out)
		output_file += "_cropped"
	display(im_out, output_file)

if len(sys.argv) < 3:
	raise Exception("Insufficient arguments")
else:
	main(sys.argv[1], sys.argv[2])
