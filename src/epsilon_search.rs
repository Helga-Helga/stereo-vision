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
#[doc="Epsilon search"]
pub mod epsilon_search {
    use super::super::crossing_out_graph::crossing_out_graph::CrossingOutGraph;
    use super::super::diffusion_graph::diffusion_graph::DiffusionGraph;
    use super::super::utils::utils::neighbor_exists;
    use super::super::utils::utils::approx_equal;

    /// Returns array of differences between
    /// * minimum vertex weight and other vertex weights in a pixel for each pixel
    /// * minimum edge weight and other edge weights between two neighbor pixels
    ///
    /// # Arguments
    /// * `crossing_out_graph`: An instance of CrossingOutGraph struct
    /// * `tolerance`: A small float value for subtracting penalties
    pub fn create_array_of_epsilons(crossing_out_graph: &mut CrossingOutGraph,
                                    tolerance: f64) -> Vec<f64> {
        let mut array: Vec<f64> = Vec::new();
        let max_disparity = crossing_out_graph.diffusion_graph.max_disparity;
        for i in 0..crossing_out_graph.diffusion_graph.left_image.len() {
            for j in 0..crossing_out_graph.diffusion_graph.left_image[0].len() {
                let min_penalty_vertex =
                    (crossing_out_graph.diffusion_graph.min_penalty_vertex(i, j)).1;
                for d in 0..max_disparity {
                    // Differences for vertices
                    if j >= d {
                        assert_ge!(crossing_out_graph.diffusion_graph
                                   .vertex_penalty_with_potentials(i, j, d), min_penalty_vertex);
                        array.push(crossing_out_graph.diffusion_graph
                                   .vertex_penalty_with_potentials(i, j, d)
                                   - min_penalty_vertex);
                        // Differences for edges
                        for n in 0..4 {
                            if neighbor_exists(i, j, n,
                                               crossing_out_graph.diffusion_graph.left_image.len(),
                                               crossing_out_graph.diffusion_graph.left_image[0].len()) {
                                let min_penalty_edge =
                                    crossing_out_graph.diffusion_graph.min_penalty_edge(i, j, n);
                                for n_d in 0..max_disparity {
                                    if crossing_out_graph.diffusion_graph.edge_exists(i, j, n, d, n_d) {
                                        assert_ge!(crossing_out_graph.diffusion_graph
                                                  .edge_penalty_with_potential(i, j, n, d, n_d),
                                                  min_penalty_edge);
                                        array.push(crossing_out_graph.diffusion_graph
                                                  .edge_penalty_with_potential(i, j, n, d, n_d)
                                                  - min_penalty_edge);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("Sorting array of epsilons ...");
        array.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("Array size: {}", array.len());
        println!("Removing duplicates from an array of epsilons ...");
        array = dedup_f64(array, tolerance);
        println!("Array size: {}", array.len());
        println!("Array of epsilons is ready");
        array
    }

    /// Returns sorted array of floats from the input, but without duplicates (with some precision)
    ///
    /// # Arguments
    /// * `array` - Sorted array of floats
    /// * `tolerance`: A small float value to compare values from array
    fn dedup_f64(array: Vec<f64>, tolerance: f64) -> Vec<f64> {
        let mut i: usize = 0;
        let mut indices_array: Vec<usize> = vec![0; array.len()];
        let mut current_index: usize = 0;
        while i < array.len() {
            indices_array[current_index] = i;
            current_index += 1;
            let mut j = i + 1;
            while j < array.len() {
                if approx_equal(array[i], array[j], tolerance) {
                    j += 1;
                } else {
                    break;
                }
            }
            i = j;
        }
        if indices_array[current_index - 1] != array.len() - 1 {
            indices_array[current_index] = array.len() - 1;
            current_index += 1;
        }
        let mut unique_values: Vec<f64> = vec![0.; current_index];
        for i in 0..current_index {
            unique_values[i] = array[indices_array[i]];
        }
        unique_values
    }

    /// Returns minimum possible epsilon from the given array which provides epsilon-consistency of a graph.
    /// It is an implementation of a binary search algorithm
    ///
    /// # Arguments:
    /// * `crossing_out_graph` - An instance of CrossingOutGraph struct
    /// * `array` - Array of all possible epsilons from `create_array_of_epsilons()`
    pub fn epsilon_search(crossing_out_graph: &mut CrossingOutGraph, array: &Vec<f64>) -> f64 {
        assert_ne!(array.len(), 0);
        let mut first_index: usize = 0;
        let mut last_index: usize = array.len() - 1;
        let mut median_index: usize = (first_index + last_index) / 2;
        while first_index < last_index {
            crossing_out_graph.initialize_with_epsilon(array[median_index]);
            crossing_out_graph.crossing_out();
            if crossing_out_graph.is_not_empty() {
                println!("Not empty for {}", median_index);
                last_index = median_index;
            } else {
                first_index = median_index + 1;
            }
            median_index = (first_index + last_index) / 2;
        }
        println!("Median_index: {}", median_index);
        array[median_index]
    }

    #[test]
    fn test_dedup_f64() {
        let mut array: Vec<f64> = [1., 1.5, 2., 3.].to_vec();
        array = dedup_f64(array, 1.);
        assert_eq!([1., 3.].to_vec(), array);
    }

    #[test]
    fn test_dedup_f64_one_element() {
        let mut array: Vec<f64> = [0.].to_vec();
        array = dedup_f64(array, 0.);
        assert_eq!([0.].to_vec(), array);
    }

    #[test]
    fn test_dedup_f64_no_remove() {
        let mut array: Vec<f64> = [1., 1.5, 2., 3.].to_vec();
        array = dedup_f64(array, 0.);
        assert_eq!([1., 1.5, 2., 3.].to_vec(), array);
    }

    #[test]
    fn test_epsilon_search_zero() {
        let left_image = vec![vec![0u32; 2]; 1];
        let right_image = vec![vec![0u32; 2]; 1];
        let max_disparity = 2;
        let diffusion_graph =
            DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
        let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph =
            CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
        let array: Vec<f64> = create_array_of_epsilons(&mut crossing_out_graph, 0.01);
        let epsilon: f64 = epsilon_search(&mut crossing_out_graph, &array);
        assert_eq!(epsilon, 0.);
    }

    #[test]
    fn test_epsilon_search() {
        let left_image = [[1, 1].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec()].to_vec();
        let max_disparity = 2;
        let mut diffusion_graph =
            DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
        diffusion_graph.potentials[0][0][2][0] = 9.4;
        diffusion_graph.potentials[0][1][0][0] = -1.8;
        diffusion_graph.potentials[0][1][0][1] = 6.7;
        let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
        let array: Vec<f64> = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.].to_vec();
        assert_eq!(9., epsilon_search(&mut crossing_out_graph, &array));
    }

    #[test]
    fn test_create_array_of_epsilons() {
        let left_image = [[1, 1].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec()].to_vec();
        let max_disparity: usize = 2;
        let mut diffusion_graph =
            DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
        diffusion_graph.potentials[0][0][2][0] = 0.6;
        diffusion_graph.potentials[0][1][0][0] = -0.3;
        diffusion_graph.potentials[0][1][0][1] = 0.1;
        let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
        let mut array: Vec<f64> = create_array_of_epsilons(&mut crossing_out_graph, 0.);
        assert_eq!(3, array.len());
        assert_eq!(0., array[0]);
        assert!(approx_equal(1.4, array[1], 1E-6));
        assert!(approx_equal(1.4, array[2], 1E-6));

        array = create_array_of_epsilons(&mut crossing_out_graph, 1.5);
        assert_eq!(2, array.len());
        assert_eq!(0., array[0]);
        assert!(approx_equal(1.4, array[1], 1E-6));
    }
}
