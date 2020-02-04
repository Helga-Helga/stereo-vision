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
#[doc="Crossing out graph"]
pub mod crossing_out_graph {
    use std::f64;
    use super::super::diffusion_graph::diffusion_graph::DiffusionGraph;
    use super::super::superpixels::superpixels::SuperpixelRepresentation;
    use super::super::utils::utils::neighbor_exists;
    use super::super::utils::utils::neighbor_index;
    use super::super::utils::utils::neighbor_superpixel;
    use super::super::utils::utils::check_disparity_map_consistency;
    use super::super::pgm_handler::pgm::pgm_writer;

    #[derive(Debug)]
    /// Crossing out graph is represented here
    pub struct CrossingOutGraph {
        /// Diffusion graph with known vertex and edge penalties
        pub diffusion_graph: DiffusionGraph,
        // A vertex is defined by superpixel coordinates `(super_i, super_j)`,
        // `superpixel` (`0` -- light or `1` -- dark) and disparity value `d`
        vertices: Vec<Vec<Vec<Vec<bool>>>>,
        // And edge is defined by a vertex `(super_i, super_j, superpixel, d)`,
        // a number of its neighbor `n` (from `0` to `8`) and disparity `n_d` of the neighbor
        edges: Vec<Vec<Vec<Vec<Vec<Vec<bool>>>>>>
    }

    impl CrossingOutGraph {
        /// Returns a crossing out graph with given parameters
        ///
        /// # Arguments
        ///
        /// * `diffusion_graph` - A DiffusionGraph object
        /// * `vertices` - A 4D vector of booleans (`false` for crossed out vertices)
        /// * `edges` - A 6D vector of booleans (`false` for crossed out edges)
        pub fn initialize(diffusion_graph: DiffusionGraph,
                          vertices: Vec<Vec<Vec<Vec<bool>>>>,
                          edges: Vec<Vec<Vec<Vec<Vec<Vec<bool>>>>>>) -> Self {
            Self {
                diffusion_graph: diffusion_graph,
                vertices: vertices,
                edges: edges
            }
        }

        /// Makes `true` only those vertices and edges in the crossing out graph,
        /// where penalties differ from the minimum penalty in the group not more than by epsilon
        ///
        /// # Arguments:
        ///
        /// * `epsilon` - A small float number for penalties comparison
        pub fn initialize_with_epsilon(&mut self, epsilon: f64) {
            self.initialize_vertices(epsilon);
            println!("Initialized vertices");
            self.initialize_edges(epsilon);
            println!("Initialized edges");
        }

        /// Makes `true` only those vertices in the crossing out graph,
        /// where penalties differ from the minimum penalty in superpixel not more than by epsilon
        ///
        /// # Arguments:
        ///
        /// * `epsilon` - A small float number for penalties comparison
        pub fn initialize_vertices(&mut self, epsilon: f64) {
            for super_i in 0..self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels {
                    for superpixel in 0..2 {
                        let min_penalty_vertex = (
                            self.diffusion_graph.min_penalty_vertex(super_i, super_j, superpixel)
                        ).1;
                        for d in 0..self.diffusion_graph.max_disparity {
                            let left_j_in_superpixel =
                                self.diffusion_graph.superpixel_representation.left_j_in_superpixel(
                                    super_i, super_j, superpixel
                            );
                            if left_j_in_superpixel >= d
                            && self.diffusion_graph.vertex_penalty_with_potentials(
                                super_i, super_j, d, superpixel
                            ) <= min_penalty_vertex + epsilon {
                                self.vertices[super_i][super_j][superpixel][d] = true;
                            } else {
                                self.vertices[super_i][super_j][superpixel][d] = false;
                            }
                        }
                    }
                }
            }
        }

        /// Makes `true` only those edges in the crossing out graph,
        /// where penalties differ from the minimum penalty from the vertex to its neighbors
        /// not more than by epsilon
        ///
        /// # Arguments:
        ///
        /// * `epsilon` - A small float number for penalties comparison
        pub fn initialize_edges(&mut self, epsilon: f64) {
            for super_i in 0..self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels {
                    for superpixel in 0..2 {
                        for n in 0..9 {
                            if neighbor_exists(super_i, super_j, n,
                                self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels,
                                self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels
                            ) {
                                let min_penalty_edge = self.diffusion_graph.min_penalty_edge(
                                    super_i, super_j, n, superpixel
                                );
                                for d in 0..self.diffusion_graph.max_disparity {
                                    for n_d in 0..self.diffusion_graph.max_disparity {
                                        if self.diffusion_graph.edge_exists(
                                            super_i, super_j, n, d, n_d, superpixel
                                        ) && self.diffusion_graph.edge_penalty_with_potential(
                                            super_i, super_j, n, d, n_d, superpixel
                                        ) <= min_penalty_edge + epsilon {
                                            self.edges[super_i][super_j][superpixel][d][n][n_d] =
                                                true;
                                        } else {
                                            self.edges[super_i][super_j][superpixel][d][n][n_d] =
                                                false;
                                        }
                                    }
                                }
                            } else {
                                for d in 0..self.diffusion_graph.max_disparity {
                                    for n_d in 0..self.diffusion_graph.max_disparity {
                                        self.edges[super_i][super_j][superpixel][d][n][n_d] = false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// Crosses out (makes `false`) a vertex, if an edge, that contains the vertex, is `false`.
        /// Crosses out an edge, if a vertex, that is connected with the edge, is `false`.
        /// Crossing out is finished when nothing was changed during one iteration
        pub fn crossing_out(&mut self) {
            let mut change_indicator = true;
            while change_indicator {
                change_indicator = false;
                for super_i in 0..self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels {
                    for super_j in 0..self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels {
                        for superpixel in 0..2 {
                            for d in 0..self.diffusion_graph.max_disparity {
                                let left_j_in_superpixel =
                                    self.diffusion_graph.superpixel_representation.left_j_in_superpixel(
                                    super_i, super_j, superpixel
                                );
                                if left_j_in_superpixel >= d {
                                    if !self.vertices[super_i][super_j][superpixel][d] {
                                        for n in 0..9 {
                                            if neighbor_exists(
                                                super_i, super_j, n,
                                                self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels,
                                                self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels
                                            ) {
                                                let (n_i, n_j, n_index) = neighbor_index(
                                                    super_i, super_j, n, superpixel
                                                );
                                                let n_superpixel = neighbor_superpixel(superpixel, n);
                                                for n_d in 0..self.diffusion_graph.max_disparity {
                                                    if self.diffusion_graph.edge_exists(
                                                        super_i, super_j, n, d, n_d, superpixel
                                                    ) && self.edges[super_i][super_j][superpixel][d][n][n_d] {
                                                        self.edges[super_i][super_j][superpixel][d][n][n_d] =
                                                            false;
                                                        self.edges[n_i][n_j][n_superpixel][n_d][n_index][d] =
                                                            false;
                                                        change_indicator = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    for n in 0..9 {
                                        if !neighbor_exists(super_i, super_j, n,
                                            self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels,
                                            self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels
                                        ) {
                                            continue;
                                        }
                                        let mut edge_exist = false;
                                        for n_d in 0..self.diffusion_graph.max_disparity {
                                            if self.diffusion_graph.edge_exists(
                                                super_i, super_j, n, d, n_d, superpixel
                                            ) && self.edges[super_i][super_j][superpixel][d][n][n_d] {
                                                edge_exist = true;
                                                break;
                                            }
                                        }
                                        if !edge_exist
                                        && self.vertices[super_i][super_j][superpixel][d] {
                                            self.vertices[super_i][super_j][superpixel][d] = false;
                                            change_indicator = true;
                                        }
                                    }
                                }
                            }
                         }
                    }
                }
            }
        }

        /// Returns `true` if there is at least one `true` vertex in each pixel
        /// and at least one `true` edge between each pair of neighbors
        pub fn is_not_empty(&self) -> bool {
            if self.vertices_exist() && self.edges_exist() {
                true
            } else {
                false
            }
        }

        /// Returns `true` if there is at least one `true` vertex in each object (superpixel) of a graph
        pub fn vertices_exist(&self) -> bool {
            for super_i in 0..self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels {
                    for superpixel in 0..2 {
                        let mut found: bool = false;
                        for d in 0..self.diffusion_graph.max_disparity {
                            let left_j_in_superpixel =
                                self.diffusion_graph.superpixel_representation.left_j_in_superpixel(
                                super_i, super_j, superpixel
                            );
                            if left_j_in_superpixel >= d
                            && self.vertices[super_i][super_j][superpixel][d] {
                                found = true;
                            }
                        }
                        if !found {
                            return false;
                        }
                    }
                }
            }
            true
        }

        /// Returns `true` if there is at least one `true` edge between each pair of neighbors in a graph
        pub fn edges_exist(&self) -> bool {
            for super_i in 0..self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels {
                    for superpixel in 0..2 {
                        for n in 0..9 {
                            if !neighbor_exists(
                                super_i, super_j, n,
                                self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels,
                                self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels
                            ) {
                                continue;
                            }
                            let mut found = false;
                            for d in 0..self.diffusion_graph.max_disparity {
                                for n_d in 0..self.diffusion_graph.max_disparity {
                                    if !self.diffusion_graph.edge_exists(
                                        super_i, super_j, n, d, n_d, superpixel
                                    ) {
                                        continue;
                                    } else {
                                        if self.edges[super_i][super_j][superpixel][d][n][n_d] {
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                                if found {
                                    break;
                                }
                            }
                            if !found {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        }

        #[cfg_attr(tarpaulin, skip)]
        /// Calls diffusion while crossing out graph is empty after crossing out with given epsilon
        ///
        /// # Arguments:
        /// * `epsilon` - A small float value for comparing penalties
        /// * `batch_size` - Number of iterations after that crossing out will be done
        pub fn diffusion_while_not_consistent(&mut self, epsilon: f64, batch_size: usize) {
            let mut not_empty: bool = false;
            let mut i = 1;
            while !not_empty {
                self.diffusion_graph.diffusion(i, batch_size);
                println!("{} iterations of diffusion ended", i + batch_size - 1);
                println!("Initializing crossing out graph with epsilon ...");
                self.initialize_with_epsilon(epsilon);
                println!("Initialized with epssilon");
                if self.is_not_empty() {
                    println!("After initialization with epsilon, resulting graph is not empty");
                    println!("Crossing out ...");
                    self.crossing_out();
                    println!("Crossing out is done");
                    not_empty = self.is_not_empty();
                    if !not_empty {
                        println!("Need more diffusion iterations");
                    }
                }
                i += batch_size;
            }
            println!("Graph is consistent. Diffusion is done");
        }

        /// Finds a vertex with the minimum penalty among not-crossed out vertexes in pixel and
        // returns a disparity correspondent to this vertex
        ///
        /// # Arguments
        ///
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        pub fn min_vertex_between_existing(&self, super_i: usize, super_j: usize,
                                           superpixel: usize) -> usize {
            let mut min_vertex: f64 = f64::INFINITY;
            let mut disparity: usize = 0;
            let mut found: bool = false;
            for d in 0..self.diffusion_graph.max_disparity {
                let left_j_in_superpixel =
                    self.diffusion_graph.superpixel_representation.left_j_in_superpixel(
                        super_i, super_j, superpixel
                    );
                if left_j_in_superpixel >= d && self.vertices[super_i][super_j][superpixel][d] {
                    let current_vertex = self.diffusion_graph.vertex_penalty_with_potentials(
                        super_i, super_j, d, superpixel
                    );
                    if current_vertex < min_vertex {
                        min_vertex = current_vertex;
                        disparity = d;
                        found = true;
                    }
                }
            }
            assert!(found, "Disparity for pixel ({}, {}) wasn't found", super_i, super_j);
            disparity
        }

        /// Chooses the highest vertex (with minimum disparity) among non-crossed in te pixel
        ///
        /// # Arguments:
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        pub fn top_vertex_between_existing(&self, super_i: usize, super_j: usize,
                                           superpixel: usize) -> usize {
            for d in 0..self.diffusion_graph.max_disparity {
                let left_j_in_superpixel =
                    self.diffusion_graph.superpixel_representation.left_j_in_superpixel(
                        super_i, super_j, superpixel
                    );
                if left_j_in_superpixel >= d && self.vertices[super_i][super_j][superpixel][d] {
                    return d;
                }
            }
            assert!(false, "Disparity for pixel ({}, {}) wasn't found", super_i, super_j);
            return 0;
        }

        /// Chooses the highest vertex (with minimum disparity) among non-crossed in each superpixel
        pub fn simple_best_labeling(&self) -> Vec<Vec<usize>> {
            let mut depth_map = vec![
                vec![vec![0usize; 2];
                     self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels];
                self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels];
            let mut depth_map_image = vec![vec![0usize; self.diffusion_graph.left_image[0].len()];
                                           self.diffusion_graph.left_image.len()];
            for super_i in 0..self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.diffusion_graph.superpixel_representation. number_of_horizontal_superpixels {
                    for superpixel in 0..2 {
                        depth_map[super_i][super_j][superpixel] = self.top_vertex_between_existing(
                            super_i, super_j, superpixel
                        );
                    }

                    for image_i in (super_i *
                            self.diffusion_graph.superpixel_representation.super_height)..(
                            super_i *
                                self.diffusion_graph.superpixel_representation.super_height +
                                self.diffusion_graph.superpixel_representation.super_height) {
                        for image_j in (super_j *
                                self.diffusion_graph.superpixel_representation.super_width)..(
                                super_j *
                                    self.diffusion_graph.superpixel_representation.super_width +
                                    self.diffusion_graph.superpixel_representation.super_width) {
                            if self.diffusion_graph.superpixel_representation.superpixels[
                                    image_i][image_j] == 0 {
                                depth_map_image[image_i][image_j] = depth_map[super_i][super_j][0];
                            } else {
                                depth_map_image[image_i][image_j] = depth_map[super_i][super_j][1];
                            }
                        }
                    }
                }
            }
            let mut depth_map_0 =
                vec![vec![0usize;
                          self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels];
                     self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels];
            let mut depth_map_1 =
                vec![
                    vec![
                        0usize;
                        self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels];
                    self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels
                ];
            for super_i in 0..self.diffusion_graph.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.diffusion_graph.superpixel_representation.number_of_horizontal_superpixels {
                    depth_map_0[super_i][super_j] = depth_map[super_i][super_j][0];
                    depth_map_1[super_i][super_j] = depth_map[super_i][super_j][1];
                }
            }
            println!(
                "Disparity map 0 is consistent: {}",
                check_disparity_map_consistency(&depth_map_0));
            println!(
                "Disparity map 1 is consistent: {}",
                check_disparity_map_consistency(&depth_map_1));
            let f = pgm_writer(
                &depth_map_0,
                "images/results/depth_map_superpixels_0.pgm".to_string(),
                self.diffusion_graph.max_disparity
            );
            let _f = match f {
                Ok(file) => file,
                Err(error) => {
                    panic!("There was a problem writing a file : {:?}", error)
                },
            };
            println!("Disparity map is saved to `depth_map_superpixels_0.pgm`");
            let f = pgm_writer(
                &depth_map_1,
                "images/results/depth_map_superpixels_1.pgm".to_string(),
                self.diffusion_graph.max_disparity
            );
            let _f = match f {
                Ok(file) => file,
                Err(error) => {
                    panic!("There was a problem writing a file : {:?}", error)
                },
            };
            println!("Disparity map is saved to `depth_map_superpixels_1.pgm`");
            depth_map_image
        }
    }
    //
    // #[test]
    // fn test_empty_crossing_out_graph() {
    //     let left_image = vec![vec![0u32; 2]; 1];
    //     let right_image = vec![vec![0u32; 2]; 1];
    //     let max_disparity = 1;
    //     let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //     let mut vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
    //     vertices[0][0][0] = false;
    //     let mut edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
    //     edges[0][0][0][2][0] = true;
    //     let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //     crossing_out_graph.crossing_out();
    //     println!("{:?}", crossing_out_graph.vertices);
    //     println!("{:?}", crossing_out_graph.edges);
    //     assert!(!crossing_out_graph.vertices_exist());
    //     assert!(!crossing_out_graph.edges_exist());
    //     assert!(!crossing_out_graph.is_not_empty());
    // }
    //
    // #[test]
    // fn test_crossing_out() {
    //     let left_image = [[1, 1].to_vec(), [1, 0].to_vec()].to_vec();
    //     let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
    //     let max_disparity: usize = 2;
    //     let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //     let vertices = vec![vec![vec![true; max_disparity]; 2]; 2];
    //     let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 2];
    //     let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //     crossing_out_graph.edges[0][0][0][2][0] = false;
    //     crossing_out_graph.edges[0][1][0][0][0] = false;
    //     crossing_out_graph.edges[0][1][1][3][1] = false;
    //     crossing_out_graph.edges[1][1][1][1][1] = false;
    //     crossing_out_graph.crossing_out();
    //     assert!(crossing_out_graph.vertices[0][0][0]);
    //     assert!(crossing_out_graph.vertices[0][1][1]);
    //     assert!(crossing_out_graph.vertices[1][0][0]);
    //     assert!(crossing_out_graph.vertices[1][1][0]);
    //     assert!(!crossing_out_graph.vertices[0][1][0]);
    //     assert!(!crossing_out_graph.vertices[1][1][1]);
    //     assert!(crossing_out_graph.edges[0][0][0][2][1]);
    //     assert!(crossing_out_graph.edges[0][0][0][3][0]);
    //     assert!(crossing_out_graph.edges[1][0][0][2][0]);
    //     assert!(!crossing_out_graph.edges[1][0][0][2][1]);
    //     assert!(crossing_out_graph.edges[0][1][1][3][0]);
    //     assert!(!crossing_out_graph.edges[0][1][1][3][1]);
    //     assert!(!crossing_out_graph.edges[0][0][0][2][0]);
    // }
    //
    #[test]
    fn test_initialize_vertices() {
        let left_image = [[2, 0].to_vec()].to_vec();
        let right_image = [[2, 0].to_vec()].to_vec();
        let max_disparity: usize = 2;
        let mut superpixel_representation = SuperpixelRepresentation::initialize(
            &left_image, 1, 2
        );
        superpixel_representation.split_into_superpixels();
        let vertical_superpixels = superpixel_representation.number_of_vertical_superpixels;
        let horizontal_superpixels = superpixel_representation.number_of_horizontal_superpixels;
        let mut diffusion_graph = DiffusionGraph::initialize(
            left_image, right_image, max_disparity, 1., superpixel_representation
        );
        diffusion_graph.potentials[0][0][1][0][0] = 0.6;
        diffusion_graph.potentials[0][0][0][0][0] = -13.7;
        diffusion_graph.potentials[0][0][0][0][1] = 80.;
        let vertices = vec![vec![vec![vec![false; max_disparity]; 2];
                                 horizontal_superpixels];
                            vertical_superpixels];
        let edges = vec![vec![vec![vec![vec![vec![false; max_disparity]; 9]; max_disparity]; 2];
                              horizontal_superpixels];
                         vertical_superpixels];
        let mut crossing_out_graph = CrossingOutGraph::initialize(
            diffusion_graph, vertices, edges
        );
        crossing_out_graph.initialize_vertices(64.);
        assert!(crossing_out_graph.vertices[0][0][1][0]);
        assert!(!crossing_out_graph.vertices[0][0][0][0]);
        assert!(crossing_out_graph.vertices[0][0][0][1]);
    }

    #[test]
    fn test_initialize_edges() {
        let left_image = [[2, 0].to_vec()].to_vec();
        let right_image = [[2, 0].to_vec()].to_vec();
        let max_disparity: usize = 2;
        let mut superpixel_representation = SuperpixelRepresentation::initialize(
            &left_image, 1, 2
        );
        superpixel_representation.split_into_superpixels();
        let vertical_superpixels = superpixel_representation.number_of_vertical_superpixels;
        let horizontal_superpixels = superpixel_representation.number_of_horizontal_superpixels;
        let mut diffusion_graph = DiffusionGraph::initialize(
            left_image, right_image, max_disparity, 1., superpixel_representation
        );
        diffusion_graph.potentials[0][0][1][0][0] = 0.6;
        diffusion_graph.potentials[0][0][0][0][0] = -13.7;
        diffusion_graph.potentials[0][0][0][0][1] = 80.;
        let vertices = vec![vec![vec![vec![false; max_disparity]; 2];
                                 horizontal_superpixels];
                            vertical_superpixels];
        let edges = vec![vec![vec![vec![vec![vec![false; max_disparity]; 9]; max_disparity]; 2];
                              horizontal_superpixels];
                         vertical_superpixels];
        let mut crossing_out_graph = CrossingOutGraph::initialize(
            diffusion_graph, vertices, edges
        );
        crossing_out_graph.initialize_edges(94.6);
        assert!(crossing_out_graph.edges[0][0][1][0][0][0]);
        assert!(!crossing_out_graph.edges[0][0][0][0][0][1]);
        assert!(crossing_out_graph.edges[0][0][0][0][0][0]);
        assert!(!crossing_out_graph.edges[0][0][0][1][0][0]);
    }
    //
    // #[test]
    // fn test_initialize_with_epsilon() {
    //     let left_image = [[1, 1].to_vec()].to_vec();
    //     let right_image = [[1, 0].to_vec()].to_vec();
    //     let max_disparity: usize = 2;
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //     diffusion_graph.potentials[0][0][2][0] = 0.6;
    //     diffusion_graph.potentials[0][1][0][0] = -0.3;
    //     diffusion_graph.potentials[0][1][0][1] = 0.1;
    //     let vertices = vec![vec![vec![false; max_disparity]; 2]; 1];
    //     let edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
    //     let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //     crossing_out_graph.initialize_with_epsilon(1.3);
    //     assert!(crossing_out_graph.vertices[0][0][0]);
    //     assert!(!crossing_out_graph.vertices[0][1][0]);
    //     assert!(crossing_out_graph.vertices[0][1][1]);
    //     assert!(crossing_out_graph.edges[0][0][0][2][0]);
    //     assert!(!crossing_out_graph.edges[0][0][0][2][1]);
    // }
    //
    // #[test]
    // fn test_vertices_exist() {
    //     let left_image = [[1, 1].to_vec()].to_vec();
    //     let right_image = [[1, 0].to_vec()].to_vec();
    //     let max_disparity: usize = 2;
    //     let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //     let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
    //     let edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
    //     let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //     crossing_out_graph.vertices[0][1][1] = false;
    //     assert!(crossing_out_graph.vertices_exist());
    // }
    //
    // #[test]
    // fn test_vertices_exist_negative() {
    //     let left_image = [[1, 1].to_vec()].to_vec();
    //     let right_image = [[1, 0].to_vec()].to_vec();
    //     let max_disparity: usize = 1;
    //     let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //     let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
    //     let edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
    //     let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //     crossing_out_graph.vertices[0][0][0] = false;
    //     assert!(!crossing_out_graph.vertices_exist());
    // }
    //
    // #[test]
    // fn test_edge_exist() {
    //     let left_image = [[1, 1].to_vec(), [1, 0].to_vec()].to_vec();
    //     let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
    //     let max_disparity: usize = 2;
    //     let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //     let vertices = vec![vec![vec![true; max_disparity]; 2]; 2];
    //     let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 2];
    //     let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //     crossing_out_graph.edges[0][0][0][2][0] = false;
    //     crossing_out_graph.edges[0][1][0][0][0] = false;
    //     crossing_out_graph.edges[0][1][0][3][1] = false;
    //     crossing_out_graph.edges[1][1][1][1][0] = false;
    //     crossing_out_graph.edges[0][1][1][3][0] = false;
    //     crossing_out_graph.edges[1][1][0][1][1] = false;
    //     crossing_out_graph.edges[0][1][1][3][1] = false;
    //     crossing_out_graph.edges[1][1][1][1][1] = false;
    //     assert!(crossing_out_graph.edges_exist());
    //     crossing_out_graph.edges[0][0][0][3][0] = false;
    //     crossing_out_graph.edges[1][0][0][1][0] = false;
    //     assert!(!crossing_out_graph.edges_exist());
    //  }
    //
    //  #[test]
    //  fn test_is_not_empty() {
    //      let left_image = [[1, 1].to_vec()].to_vec();
    //      let right_image = [[1, 0].to_vec()].to_vec();
    //      let max_disparity: usize = 2;
    //      let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //      let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
    //      let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
    //      let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //      crossing_out_graph.vertices[0][1][0] = false;
    //      crossing_out_graph.edges[0][0][0][2][1] = false;
    //      crossing_out_graph.edges[0][1][1][0][0] = false;
    //      assert!(crossing_out_graph.is_not_empty());
    //      crossing_out_graph.vertices[0][1][0] = true;
    //      crossing_out_graph.vertices[0][0][0] = false;
    //      assert!(!crossing_out_graph.is_not_empty());
    //      crossing_out_graph.vertices[0][0][0] = true;
    //      crossing_out_graph.edges[0][0][0][2][0] = false;
    //      crossing_out_graph.edges[0][1][0][0][0] = false;
    //      assert!(!crossing_out_graph.is_not_empty());
    //  }
    //
    //  #[test]
    //  fn test_min_vertex_between_existing() {
    //      let left_image = [[1, 1, 0].to_vec()].to_vec();
    //      let right_image = [[1, 0, 0].to_vec()].to_vec();
    //      let max_disparity: usize = 3;
    //      let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //      diffusion_graph.potentials[0][0][2][0] = 0.6;
    //      diffusion_graph.potentials[0][1][0][0] = -13.7;
    //      diffusion_graph.potentials[0][1][0][1] = 80.;
    //      diffusion_graph.potentials[0][1][2][0] = 358.;
    //      diffusion_graph.potentials[0][1][2][1] = -1E9;
    //      diffusion_graph.potentials[0][2][0][0] = -0.3;
    //      diffusion_graph.potentials[0][2][0][1] = 0.1;
    //      diffusion_graph.potentials[0][2][0][2] = 0.8;
    //      let vertices = vec![vec![vec![true; max_disparity]; 3]; 1];
    //      let edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 3]; 1];
    //      let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //      assert_eq!(0, crossing_out_graph.min_vertex_between_existing(0, 0));
    //      crossing_out_graph.vertices[0][1][1] = false;
    //      assert_eq!(0, crossing_out_graph.min_vertex_between_existing(0, 1));
    //      crossing_out_graph.vertices[0][2][1] = false;
    //      assert_eq!(2, crossing_out_graph.min_vertex_between_existing(0, 2));
    //  }
    //
    //  #[test]
    //  #[should_panic]
    //  fn test_min_vertex_between_existing_empty() {
    //      let left_image = [[1].to_vec()].to_vec();
    //      let right_image = [[1].to_vec()].to_vec();
    //      let max_disparity:usize = 0;
    //      let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //      let vertices = vec![vec![vec![true; max_disparity]; 1]; 1];
    //      let edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 1]; 1];
    //      let crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //      crossing_out_graph.min_vertex_between_existing(0, 0);
    //  }
    //
    //  #[test]
    //  #[should_panic]
    //  fn test_cross_vertex_that_is_crossed() {
    //      let left_image = [[1, 1].to_vec()].to_vec();
    //      let right_image = [[1, 0].to_vec()].to_vec();
    //      let max_disparity: usize = 2;
    //      let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //      let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
    //      let edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
    //      let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //      crossing_out_graph.vertices[0][1][0] = false;
    //      crossing_out_graph.cross_vertex(0, 1, 0);
    //  }
    //
    //  #[test]
    //  fn test_cross_vertex_one_disparity() {
    //      let left_image = [[1].to_vec()].to_vec();
    //      let right_image = [[1].to_vec()].to_vec();
    //      let max_disparity:usize = 1;
    //      let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //      let vertices = vec![vec![vec![true; max_disparity]; 1]; 1];
    //      let edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 1]; 1];
    //      let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //      crossing_out_graph.cross_vertex(0, 0, 0);
    //      assert!(crossing_out_graph.vertices[0][0][0]);
    //  }
    //
    //  #[test]
    //  fn test_cross_vertex_two_disparities() {
    //      let left_image = [[1, 1].to_vec()].to_vec();
    //      let right_image = [[1, 0].to_vec()].to_vec();
    //      let max_disparity: usize = 2;
    //      let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //      let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
    //      let edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
    //      let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //      crossing_out_graph.cross_vertex(0, 1, 0);
    //      assert!(crossing_out_graph.vertices[0][1][0]);
    //      assert!(!crossing_out_graph.vertices[0][1][1]);
    //  }
    //
    //  #[test]
    //  fn test_find_best_labeling() {
    //      let left_image = [[1, 1, 0].to_vec()].to_vec();
    //      let right_image = [[1, 0, 0].to_vec()].to_vec();
    //      let max_disparity: usize = 3;
    //      let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, max_disparity, 1.);
    //      diffusion_graph.potentials[0][0][2][0] = 0.6;
    //      diffusion_graph.potentials[0][1][0][0] = -13.7;
    //      diffusion_graph.potentials[0][1][0][1] = 80.;
    //      diffusion_graph.potentials[0][1][2][0] = 358.;
    //      diffusion_graph.potentials[0][1][2][1] = -1E9;
    //      diffusion_graph.potentials[0][2][0][0] = -0.3;
    //      diffusion_graph.potentials[0][2][0][1] = 0.1;
    //      diffusion_graph.potentials[0][2][0][2] = 0.8;
    //      let vertices = vec![vec![vec![true; max_disparity]; 3]; 1];
    //      let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 3]; 1];
    //      let mut crossing_out_graph = CrossingOutGraph::initialize(diffusion_graph, vertices, edges);
    //      let disparity_map = crossing_out_graph.find_best_labeling();
    //      assert_eq!(0, disparity_map[0][0]);
    //      assert_eq!(0, disparity_map[0][1]);
    //      assert_eq!(1, disparity_map[0][2]);
    //  }
}
