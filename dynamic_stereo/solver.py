from numpy import array, zeros, append, uint8
from PIL import Image
from matplotlib import pyplot
from graph import initialize_graph, fill_nodes, fill_edges
from Semiring.SemiringArgminPlusELement import SemiringArgminPlusElement


def dynamic_programming_solver(nodes, edges,
                               semiring=SemiringArgminPlusElement):
    """
    Dynamic programming algorithm for finding the best path
    (with minimum weight)
    :param nodes: 3d array of node weights
    :param edges: 3d array of edge weights
    :param semiring: semiring class to use
    :return: 3d array of updated node weights
    """
    for obj in reversed(range(1, nodes.shape[0])):
        for label_l in range(nodes.shape[1]):
            min_value = semiring.get_zero()
            next_label = None
            for label_r in range(nodes.shape[2]):
                min_value, next_label = \
                    min_value.add(edges[obj-1, label_l, label_r]
                                  .mul(nodes[obj, label_r, 0]),
                                  next_label, label_r)
            nodes[obj-1, label_l, 0] = nodes[obj-1, label_l, 0].mul(min_value)
            nodes[obj-1, label_l, 1] = next_label
    return nodes


def get_best_path(nodes, semiring=SemiringArgminPlusElement):
    """
    Finds path with minimum weight
    :param nodes: 3d array of updated node weights after dynamic programming
    :param semiring: semiring class to use
    :return: best path presented as a list of consecutive labels and its weight
    """
    path = [0] * nodes.shape[0]
    min_value = semiring.get_zero()
    for label in range(nodes.shape[1]):
        if nodes[0, label, 0].value < min_value.value:
            min_value = nodes[0, label, 0]
            path[0] = label
    for obj in range(1, nodes.shape[0]):
        path[obj] = int(nodes[obj-1, path[obj-1], 1])
    return [i for i in path], nodes[0, path[0], 0]

if __name__ == "__main__":
    MAX_DISPARITY = 3
    left_image = array(Image.open("../images/left.png").convert("L"),
                       dtype=int)
    pyplot.imshow(left_image, 'gray')
    print(left_image)
    # pyplot.show()
    right_image = array(Image.open("../images/right.png").convert("L"),
                        dtype=int)
    pyplot.imshow(right_image, 'gray')
    print(right_image)
    # pyplot.show()

    disparity_map = zeros(left_image.shape, dtype=uint8)

    for i in range(left_image.shape[0]):
        nodes, edges = initialize_graph(left_image.shape[1], MAX_DISPARITY)
        nodes = fill_nodes(nodes, left_image, right_image, i)
        edges = fill_edges(edges)

        nodes = dynamic_programming_solver(nodes, edges)
        path, path_weight = get_best_path(nodes)
        disparity_map[i] = path
        print("Line {} : result path {} : path weight {}"
              .format(i, path, path_weight))

    print(disparity_map)
    pyplot.imshow(disparity_map, 'gray')
    # pyplot.show()
