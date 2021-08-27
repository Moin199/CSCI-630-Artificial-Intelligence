import math
import sys
from collections import deque as queue
from PIL import Image
from queue import PriorityQueue

Distance = []
terrain_dictionary = {(248, 148, 18): ("Open Land", 5.5), (255, 192, 0): ("Rough Meadow", 11),
                      (255, 255, 255): ("Easy movement forest", 7), (2, 208, 60): ("Slow run forest", 9),
                      (2, 136, 40): ("Walk forest", 6), (5, 73, 24): ("Imppassible Vegetation", float("inf")),
                      (71, 51, 3): ("Paved road", 4), (0, 0, 255): ("Lake;Swamp;Marsh", float("inf")),
                      (0, 0, 0): ("Footpath", 3), (205, 0, 101): ("Out of bounds", float("inf")),
                      (0, 100, 255): ("Mud", 1000), (0, 255, 255): ("Frost", 1000)}


class Node:
    __slots__ = "x", "y", "terrain", "elevation", "gn", "fn", "parent"

    def __init__(self, x, y, terrain, elevation, gn=0, fn=0, parent=None):
        self.x = x
        self.y = y
        self.terrain = terrain
        self.elevation = elevation
        self.gn = gn
        self.fn = fn
        self.parent = parent

    def __lt__(self, other):
        return self.fn < other.fn

    def getCoordinates(self):
        return self.x, self.y


def get2dpixelArray(image, elevation):
    im = Image.open(image)
    im1 = im.load()
    img_width, img_height = im.size
    image_array = []
    elevation_array = []
    with open(elevation) as f:
        elevation = f.readlines()
    for i in range(img_height):
        elevation_array.append(elevation[i].split()[:395])
    for x in range(img_width):
        temp = []
        for y in range(img_height):
            temp.append(Node(x, y, terrain_dictionary[im1[x, y][:3]], elevation_array[y][x]))
        image_array.append(temp)
    return image_array


def get_neighbour(node, image_array):
    neighbor = []
    x = int(node.x)
    y = int(node.y)
    if x + 1 < 395:
        neighbor.append(image_array[x + 1][y])

    if x - 1 > 0:
        neighbor.append(image_array[x - 1][y])
    if y + 1 < 500:
        neighbor.append(image_array[x][y + 1])
    if y - 1 > 0:
        neighbor.append(image_array[x][y - 1])
    return neighbor


def calculate_heuristic(node1, node2):
    deltax = (int(node1.x) - int(node2.x)) * 10.29
    deltay = (int(node1.y) - int(node2.y)) * 7.55
    deltaz = float(node1.elevation) - float(node2.elevation)
    return math.sqrt(deltax ** 2 + deltay ** 2 + deltaz ** 2)


def calculate_gn(node1, node2):
    deltax = (int(node1.x) - int(node2.x)) * 10.29
    deltay = (int(node1.y) - int(node2.y)) * 7.55
    deltaz = float(node1.elevation) - float(node2.elevation)
    dist = math.sqrt(deltax ** 2 + deltay ** 2 + deltaz ** 2)

    return dist * node1.terrain[1]


def get_distance(node1, node2):
    deltax = (int(node1.x) - int(node2.x)) * 10.29
    deltay = (int(node1.y) - int(node2.y)) * 7.55
    deltaz = float(node1.elevation) - float(node2.elevation)
    dist = math.sqrt(deltax ** 2 + deltay ** 2 + deltaz ** 2)

    Distance.append(dist)


def astar(start, goal, image_array):
    priority_queue = PriorityQueue()
    visited = []
    path = []
    current = start
    h_n_start = calculate_heuristic(current, goal)

    gn = 0
    current.fn = gn + h_n_start

    while ((current.x != goal.x or current.y != goal.y)):

        for vertex in get_neighbour(current, image_array):

            if vertex not in visited and vertex not in priority_queue.queue:

                gn = current.gn + calculate_gn(current, vertex)
                h_n = calculate_heuristic(vertex, goal)
                f_n = gn + h_n

                if f_n < vertex.fn or vertex.fn == 0:
                    get_distance(current, vertex)
                    vertex.fn = f_n
                    vertex.parent = current
                    visited.append(vertex)
                    priority_queue.put(vertex)

        visited.append(current)
        current = priority_queue.get()
    temp = current
    if current.x == goal.x and current.y == goal.y:
        while (temp.parent is not None):
            path.append((temp.x, temp.y))
            temp = temp.parent
    path.append((temp.x, temp.y))
    return path


def get_full_path(path_file_name, image_array):
    path_array = []
    global_path = []
    with open(path_file_name) as f:
        path_file = f.readlines()
        for coordinate in path_file:
            single_coordinate = coordinate.replace("\n", "").split(" ")
            path_array.append(single_coordinate)
    for j in range(len(path_array) - 1):
        x1 = int(path_array[j][0])
        y1 = int(path_array[j][1])
        x2 = int(path_array[j + 1][0])
        y2 = int(path_array[j + 1][1])
        global_path.append(astar(image_array[x1][y1],
                                 image_array[x2][y2], image_array))
        reset_matrix(image_array)
    return global_path


def reset_matrix(image_array):
    for i in range(len(image_array)):
        for j in range(len(image_array)):
            image_array[i][j].fn = 0
            image_array[i][j].parent = None


def display_image_with_path(total_path, im, output_file_name):
    im = im

    for i in total_path:
        for j in i:
            im.putpixel((j[0], j[1]), (55, 42, 61, 110))
    distance = 0
    for i in Distance:
        if i != float("inf"):
            distance += i
    print("Total Distance Traversed:" + str(distance) + " Metres")
    im.show()
    im.save(output_file_name)


def check_visited(visited, row, col, max_row, max_col, node):
    if row < 0 or col < 0 or row >= max_row or col >= max_col:
        return False

    if (node in visited):
        return False

    return True


def bfs_traversal_for_spring(image_array, borders):
    point_to_color = []
    q = queue()
    visited = set()
    x, y = borders
    q.append((x, y))
    visited.add(image_array[x][y])
    count = 0
    while (abs(x - borders[0]) < 15 and abs(y - borders[1]) < 15) and len(q) != 0:
        cell = q.popleft()
        x = cell[0]
        y = cell[1]
        point_to_color.append((x, y))
        count += 1
        neighbour = get_neighbour(image_array[x][y], image_array)
        for i in neighbour:
            if i.terrain[0] == "Lake;Swamp;Marsh":
                continue
            adjx = i.x
            adjy = i.y
            if (check_visited(visited, adjx, adjy, len(image_array), len(image_array[0]), image_array[adjx][adjy])):
                if abs(float(image_array[x][y].elevation) - float(i.elevation)) < 1.0 and i.terrain[
                    0] != "Out of bounds":
                    q.append((adjx, adjy))
                    visited.add(image_array[adjx][adjy])
    return point_to_color


def bfs_traversal_for_winter(image_array, border_tuple):
    point_to_color = []
    q = queue()
    visited = set()
    x, y = border_tuple
    q.append((x, y))
    visited.add(image_array[x][y])
    count = 0
    while (abs(x - border_tuple[0]) < 7 and abs(y - border_tuple[1]) < 7) and len(q) != 0:
        cell = q.popleft()
        x = cell[0]
        y = cell[1]
        point_to_color.append((x, y))
        count += 1
        neighbour = get_neighbour(image_array[x][y], image_array)
        for i in neighbour:
            adjx = i.x
            adjy = i.y
            if (check_visited(visited, adjx, adjy, len(image_array), len(image_array[0]), image_array[adjx][adjy])):
                if i.terrain[0] == "Lake;Swamp;Marsh":
                    q.append((adjx, adjy))
                    visited.add(image_array[adjx][adjy])
    return point_to_color


def season_modifier(season, image_array, im):
    if season == "spring":
        color = []
        water_cordinates = set()
        for i in range(len(image_array)):
            for j in range(len(image_array[0])):
                if image_array[i][j].terrain[0] == "Lake;Swamp;Marsh":
                    water_cordinates.add(image_array[i][j])
        water_borders = set()
        for i in water_cordinates:
            neighbour = get_neighbour(i, image_array)
            for vertex in neighbour:
                if vertex.terrain[0] != "Lake;Swamp;Marsh":
                    water_borders.add((vertex.x, vertex.y))
        for border_tuple in water_borders:
            color.append(bfs_traversal_for_spring(image_array, border_tuple))
        for color_tuple in color:
            for j in color_tuple:
                im.putpixel(j, (210, 105, 30))
        return im
    elif season == "winter":
        color = []
        water_cordinates = set()
        for i in range(len(image_array)):
            for j in range(len(image_array[0])):
                if image_array[i][j].terrain[0] == "Lake;Swamp;Marsh":
                    water_cordinates.add(image_array[i][j])
        water_borders = set()
        for i in water_cordinates:
            neighbour = get_neighbour(i, image_array)
            for vertex in neighbour:
                if vertex.terrain[0] != "Lake;Swamp;Marsh":
                    water_borders.add((vertex.x, vertex.y))
        for border_tuple in water_borders:
            color.append(bfs_traversal_for_winter(image_array, border_tuple))
        for color_tuple in color:
            for j in color_tuple:
                im.putpixel(j, (0, 255, 255))
        return im
    elif season == "fall":
        for i in range(len(image_array)):
            for j in range(len(image_array[0])):
                if (image_array[i][j].terrain[0] == "Easy movement forest"):
                    updated_terrain = ("Easy movement forest", 50)
                    image_array[i][j].terrain = updated_terrain
                    # print(i,j)

                    for vertex in get_neighbour(image_array[i][j], image_array):
                        weight = 50
                        neighbour_terrain = (vertex.terrain[0], weight)
                        vertex.terrain = neighbour_terrain
        return im
    elif season.lower() == "summer":
        return im


def main():
    image_file = sys.argv[1]
    elevation_file = sys.argv[2]
    image = Image.open(image_file)
    image_array = get2dpixelArray(image_file, elevation_file)
    image_up = season_modifier(sys.argv[4], image_array, image)
    path_file = sys.argv[3]
    total_path = get_full_path(path_file, image_array)
    output_file_name = sys.argv[5]
    display_image_with_path(total_path, image_up, output_file_name)


if __name__ == '__main__':
    main()
