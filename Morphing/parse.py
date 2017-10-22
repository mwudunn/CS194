import sys
import matplotlib.pyplot as plt
import pickle
import numpy as np

def main():
	location = "./average_pts/"
	filename = "b.pts"
	for i in range(1,101):
		with open(location + str(i) + filename) as file:
			lines = []
			for line in file:
				lines.append(line)
			lines = lines[3:len(lines) - 1]

			points = []
			for each in lines:
				each = each.split()
				each[0] = float(each[0])
				each[1] = float(each[1])
				points.append(each)
			pickle.dump(np.array(points), open(location + str(i) + "b_points.txt", "wb"))
main()