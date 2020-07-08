# -------------------------------------------------------
# Assignment: 1
# Written by: Mushfiqur Anik, 40025703
# For COMP 472 Section: JX – Summer 2020
# -------------------------------------------------------

# Install the following
# Geopandas and pyshp and scipy using:
# pip install geopandas
# pip install pyshp
# pip install scipy
# Import geopandas and pyshp

# Import all the following files
import numpy as np
import matplotlib.pyplot as plt
import shapefile
from matplotlib.collections import LineCollection
import math
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import heapq
from matplotlib.pyplot import figure
from scipy.spatial import distance
import time

# Starting Point and Ending Point
startPoint = (1, 1)
endPoint = (17, 17)

# Reading the shapefile
shape = shapefile.Reader("crime_dt.shp", encoding='ISO-8859-1')
shapes = shape.shapes()

# Extracting the X and Y Coordinates from shapefile
# To plot on the grid
coordinatesList = []

i = 0
while i < len(shape):
    coordinatesList.append(shapes[i].points)
    i = i + 1

coordinatesArray = np.array(coordinatesList)

xCoordinate = coordinatesArray[:, 0, 0]
xCoordinate = xCoordinate.tolist()

yCoordinate = coordinatesArray[:, 0, 1]
yCoordinate = yCoordinate.tolist()


# This is the main loop of the project
# Asks for threshold and gridsize from user
# Calculates the shortest path from startPoint to endPoint
def mainLoop():
    # Prompting user for gridsize and threshold
    gridSize = input("Please enter the gridSize: ")
    threshold = input("Please enter the threshold: ")
    gridSize = float(gridSize)
    threshold = int(threshold)

    # Calculating number of rows and columns for grid
    rows = math.ceil((-73.55 + 73.59) / gridSize)
    cols = math.ceil((45.53 - 45.49) / gridSize)

    # Bins and weights for the grid
    binsX = []
    binsY = []
    weights = []

    i = float(-73.590)
    while i <= -73.550:
        binsX.append(i)
        i += gridSize

    j = float(45.490)
    while j <= 45.530:
        binsY.append(j)
        j += gridSize

    k = 0
    while k < 19010:
        weights.append(1)
        k += 1

    # Defining own colourmap
    cmap = mpl.colors.ListedColormap(['yellow', 'black'])

    # Looping through the graph and displaying
    graph = plt.hist2d(xCoordinate, yCoordinate, bins=[binsX, binsY], weights=weights, cmap=cmap)
    plt.show()

    # Generating new grid with ones and zeros (1 being the blocked areas)
    x = rows + 1
    y = cols + 1
    length = (len(graph[0]))
    grid = np.array([[0] * x] * y)

    # Calculating threshold
    listOfCrimeRates = []

    for i in range(0, length - 1):
        for j in range(0, length - 1):
            listOfCrimeRates.append(float(graph[0][i][j]))

    print("Crime rates list in descending order: ")
    listOfCrimeRates.sort(reverse=True)
    print(listOfCrimeRates)

    crimeRateAreas = 311
    print("Number of crime rate areas: ")
    print(crimeRateAreas)

    threshold = (crimeRateAreas * (1 - (threshold / 100)))
    threshold = int(threshold)

    print("Printing threshold index: ")
    print(threshold)

    print("Threshold index value: ")
    valueAtThreshold = listOfCrimeRates[threshold]
    print(valueAtThreshold)

    # Calculate Average and standard deviation
    print('Mean:', np.mean(listOfCrimeRates))
    print('Standard Deviation:', np.std(listOfCrimeRates))

    # Looping through the graph
    # Marking the cells greater than threshold as 1
    for i in range(0, length - 1):
        for j in range(0, length - 1):
            if graph[0][i][j] >= valueAtThreshold:

                grid[i][j] = 1
            else:
                grid[i][j] = 0

    grid = grid.transpose()
    grid = grid[::-1]

    # Calculating the path from startPoint to endPoint
    start = time.process_time()  # Start timer
    path = aStarAlgorithm(startPoint, endPoint, grid)
    print("Time taken for the algorithm to run: ")
    print(time.process_time() - start)

    if len(path) == 0:
        print("There is no path due to blocks")
    else:
        path = path + [startPoint]
        path = path[::-1]

        # X and Y coordinates from path
        xCoord = []
        yCoord = []

        for i in (range(0, len(path))):
            x = path[i][0]
            y = path[i][1]
            xCoord.append(x)
            yCoord.append(y)

        # Plot the graph
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.axis((0, 19, 0, 19))
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        ax.imshow(grid, cmap=cmap)
        ax.scatter(startPoint[1], startPoint[0], marker="8", color="red", s=200)
        ax.scatter(endPoint[1], endPoint[0], marker="D", color="red", s=200)
        ax.plot(yCoord, xCoord, color="black")
        plt.show()


# Heuristic Function using euclaidan distance
def heuristic(startPoint, endPoint):
    dist = distance.euclidean(startPoint, endPoint)
    return dist


# Main loop
# Takes in the startPoint, endPoint, and grid with 1s and 0s
def aStarAlgorithm(startPoint, endPoint, grid):
    # Neighbours
    neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # Operators/moves
    openList = []
    closeList = set()
    previousNodes = {}

    gScore = {startPoint: 0}
    fScore = {startPoint: heuristic(startPoint, endPoint)}

    heapq.heappush(openList, (fScore[startPoint], startPoint))

    # We keep looping until openList is empty
    # We have visited all the positions
    while openList:

        # We pop the neighbour with the shortest fScore
        current = heapq.heappop(openList)[1]

        # If we reached our endPoint we backtrack
        # We list all the previous nodes until we reach startPoint
        if current == endPoint:

            backTrack = []

            while current in previousNodes:
                backTrack.append(current)
                current = previousNodes[current]
            return backTrack

        # We then add this node to the closeList
        closeList.add(current)

        # If we haven't reached endPoint
        for i, j in neighbours:

            # Select one of the 8 neighbours
            neighbour = current[0] + i, current[1] + j

            # If the neighbour is out of bounds therefore avoid it
            if 0 <= neighbour[0] < grid.shape[0]:
                if 0 <= neighbour[1] < grid.shape[1]:
                    if grid[neighbour[0]][neighbour[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            # Default gScore value
            gScoreValue = 1

            # If the neighbour is diagonal its score is 1.5
            if ((i == 1 and j == 1) or (i == 1 and j == -1) or (i == -1 and j == 1) or (i == -1 and j == -1)):
                gScoreValue = 1.5

            # This is the tentative g score value from start to neighbour
            tentative_g_score = gScore[current] + gScoreValue  # 1 for up/down/left/right 1.5 for diagonal

            # Check if gScore is less than tentative_g_score and included inside closeList
            # Continue
            if gScore.get(neighbour, 0) <= tentative_g_score and neighbour in closeList:
                continue

            # If neighbour not in open List
            # Calculate it's fScore and add it to the open list
            if neighbour not in [i[1] for i in openList]:
                previousNodes[neighbour] = current
                gScore[neighbour] = tentative_g_score
                fScore[neighbour] = tentative_g_score + heuristic(neighbour, endPoint)
                heapq.heappush(openList, (fScore[neighbour], neighbour))

            # If we find a better tentative_g_score we use the new value
            if tentative_g_score < gScore.get(neighbour, 0):
                previousNodes[neighbour] = current
                gScore[neighbour] = tentative_g_score
                fScore[neighbour] = tentative_g_score + heuristic(neighbour, endPoint)
                heapq.heappush(openList, (fScore[neighbour], neighbour))

    # No path found due to blocks
    # Return empty list
    emptyList = []
    return emptyList


mainLoop()  # Start the main loop

print("Program is done!!!")