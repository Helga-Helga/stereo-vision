/*
 * MIT License
 *
 * Copyright (c) 2018 Olga Laviagina
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
pub mod epsilon_search {
    use super::super::crossing_out_graph::crossing_out_graph::CrossingOutGraph;
    use super::super::penalty_graph::penalty_graph::PenaltyGraph;
    use super::super::diffusion::diffusion::neighbour_exists;
    use super::super::diffusion::diffusion::neighbour_index;

    pub fn create_array_of_epsilons(crossing_out_graph: &mut CrossingOutGraph,
                                    tolerance: f64) -> Vec<f64> {
    /*
    crossing_out_graph: CrossingOutGraph
    Returns array of differences between
    - minimum vertex weight and other vertex weights in a pixel for each pixel and
    - minimum edge weight and other edge weights between two neighbour pixels
    */
        let mut array: Vec<f64> = Vec::new();
        let max_disparity = crossing_out_graph.penalty_graph.max_disparity;
        for i in 0..crossing_out_graph.penalty_graph.left_image.len() {
            for j in 0..crossing_out_graph.penalty_graph.left_image[0].len() {
                let min_penalty_vertex =
                    crossing_out_graph.penalty_graph.min_penalty_vertex(i, j);
                for d in 0..max_disparity {
                    // Differences for vertices
                    if j >= d {
                        array.push(crossing_out_graph.penalty_graph
                            .lookup_table[crossing_out_graph.penalty_graph.left_image[i][j]
                            as usize][crossing_out_graph.penalty_graph.right_image[i][j-d] as usize]
                            + crossing_out_graph.penalty_graph.sum_of_potentials(i, j, d)
                            - min_penalty_vertex as f64);
                    }
                    // Differences for edges
                    for n in 0..4 {
                        if neighbour_exists(i, j, n,
                                            crossing_out_graph.penalty_graph.left_image.len(),
                                            crossing_out_graph.penalty_graph.left_image[0].len()) {
                            let (n_i, n_j, n_index) = neighbour_index(i, j, n);
                            let min_penalty_edge =
                            crossing_out_graph.penalty_graph
                            .min_penalty_edge(i, j, n, n_i, n_j, n_index);
                            for n_d in 0..max_disparity {
                                array.push(crossing_out_graph.penalty_graph.lookup_table[d][n_d]
                                    - crossing_out_graph.penalty_graph.potentials[i][j][n][d]
                                    - crossing_out_graph.penalty_graph.potentials[n_i][n_j][n_index][n_d]
                                    - min_penalty_edge);
                            }
                        }
                    }
                }
            }
        }
        array.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dedup_f64(&mut array, tolerance);
        array
    }

    fn dedup_f64(array: &mut Vec<f64>, tolerance: f64){
    /*
    array: sorted array of floats
    tolerance: if array[i + 1] of array differs from array[i] less than by tolerance,
    then array[i + 1] is removed from array
    tolerance is the biggest possible value,
    with which two elements of array can be considered as equal
    Returns input sorted array of floats, but without duplicates (with some precision)
    */
        for i in 0..array.len() {
            if i + 1 < array.len() {
                if array[i + 1] - array[i] <= tolerance {
                    array.remove(i + 1);
                    dedup_f64(array, tolerance);
                } else {
                    continue;
                }
            } else {
                break;
            }
        }
    }

    fn median_index(array: &Vec<f64>) -> usize {
    /*
    array: sorted array without duplicates
    Returns an index of array median
    */
        array.len() / 2
    }

    pub fn epsilon_search(mut crossing_out_graph: CrossingOutGraph, array: &Vec<f64>,
                              first_index: usize, last_index: usize) -> f64 {
    /*
    crossing_out_graph: CrossingOutGraph
    array: array of all possible epsilons from `create_array_of_epsilons()`
    first_index: first index of current sub-array
    last_index: last index of current sub-array
    Returns minumum possible epsilon which provides epsilon-consistency
    It is recursive binary search implementation
    */
        let current_array: Vec<f64> = array[first_index..last_index].to_vec();
        let epsilon: f64 = current_array[median_index(&current_array)];
        crossing_out_graph.initialize_with_epsilon(epsilon);
        crossing_out_graph.crossing_out();
        if last_index - first_index > 1 {
            if crossing_out_graph.is_not_empty() {
                return epsilon_search(crossing_out_graph, array,
                                      first_index, median_index(&current_array));
            } else {
                return epsilon_search(crossing_out_graph, array,
                                      median_index(&current_array), last_index);
            }
        } else {
            if crossing_out_graph.is_not_empty() {
                return array[first_index];
            } else {
                return array[first_index + 1];
            }
        }
    }

    #[test]
    fn test_dedup_f64() {
        let mut array = [1., 1.5, 2., 3.].to_vec();
        dedup_f64(&mut array, 1.);
        assert_eq!([1., 3.].to_vec(), array);
    }

    #[test]
    fn test_dedup_f64_one_element() {
        let mut array = [0.].to_vec();
        dedup_f64(&mut array, 0.);
        assert_eq!([0.].to_vec(), array);
    }

    #[test]
    fn test_dedup_f64_no_remove() {
        let mut array = [1., 1.5, 2., 3.].to_vec();
        dedup_f64(&mut array, 0.);
        assert_eq!([1., 1.5, 2., 3.].to_vec(), array);
    }

    #[test]
    fn test_search_for_epsilon() {
        let left_image = vec![vec![0u32; 2]; 1];
        let right_image = vec![vec![0u32; 2]; 1];
        let max_disparity = 2;
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity);
        let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        let array: Vec<f64> = create_array_of_epsilons(&mut crossing_out_graph, 0.01);
        let epsilon: f64 = epsilon_search(crossing_out_graph, &array, 0, array.len() - 1);
        assert_eq!(epsilon, 0.0);
    }
}
