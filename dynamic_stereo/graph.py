from numpy import full, array
from PIL import Image
from Semiring.SemiringArgminPlusELement import SemiringArgminPlusElement


def initialize_graph(n_obj, max_disparity, semiring=SemiringArgminPlusElement):
    """
    Initializes nodes and edges with infinity values.
    :param n_obj: number of objects
    :param max_disparity: number of labels

    n_obj * max_disparity -- number of nodes

    Weight of an edge is its length.
    Edges go only between neighbour objects (from left to right)

    :param semiring: defines generalized operations add and mul
    :return: 3d array of nodes and 3d array of edges.
    Example of indexed node and edge: (i, k, k'),
    where i is object, k is a label in this object, k' is a label in the next object (i + 1)
    """
    nodes = full((n_obj, max_disparity, max_disparity), semiring.get_zero())
    edges = full((n_obj, max_disparity, max_disparity), semiring.get_zero())
    return nodes, edges


def fill_nodes(nodes, left_image, right_image, line):
    """

    Label is a disparity of pixel (shift in pixels to left for left image).
    Weight of a node is difference of intensities between left and right images.

    nodes[obj, disparity, 0] is weight of node (obj, disparity)
    nodes[obj, disparity, 1] is a best label in next (obj + 1) object



    :param nodes:
    :param left_image:
    :param right_image:
    :param line:
    :return:
    """
    for obj in range(nodes.shape[0]):
        for disparity in range(nodes.shape[1]):
            if obj - disparity >= 0:
                print(left_image[obj, line] - right_image[obj-disparity, line])
                nodes[obj, disparity, 0] = abs(left_image[obj, line] - right_image[obj-disparity, line])
            nodes[obj, disparity, 1] = None
    return nodes


def fill_edges(edges):
    """

    d(i+1) <= d(i) + 1 for all i

    :param edges:
    :return:
    """
    for obj in range(edges.shape[0]):
        for disparity in range(edges.shape[1]):
            for disparity_next in range(edges.shape[2]):
                if disparity_next > disparity + 1:
                    continue
                edges[obj, disparity, disparity_next] = abs(disparity - disparity_next)
    return edges


if __name__ == "__main__":
    MAX_DISPARITY = 3
    left_image = array(Image.open("../images/left3.png").convert("L"), dty)
    right_image = array(Image.open("../images/right3.png").convert("L"))

    nodes, edges = initialize_graph(left_image.shape[0], MAX_DISPARITY)
    nodes = fill_nodes(nodes, left_image, right_image, 0)
