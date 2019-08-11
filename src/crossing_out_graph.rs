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
pub mod crossing_out_graph {
    use std::f64;
    use super::super::penalty_graph::penalty_graph::PenaltyGraph;
    use super::super::diffusion::diffusion::neighbor_exists;
    use super::super::diffusion::diffusion::neighbor_index;

    #[derive(Debug)]
    pub struct CrossingOutGraph {
        pub penalty_graph: PenaltyGraph,
        // A vertice is defined by pixel coordinates (i, j) and disparity value d
        vertices: Vec<Vec<Vec<bool>>>,
        // And edge is defined by a vertex (i, j, d),
        // a number of its neighbor n (from 0 to 3) and disparity n_d of the neighbor
        edges: Vec<Vec<Vec<Vec<Vec<bool>>>>>
    }

    impl CrossingOutGraph {
        pub fn initialize(penalty_graph: PenaltyGraph,
                          vertices: Vec<Vec<Vec<bool>>>,
                          edges: Vec<Vec<Vec<Vec<Vec<bool>>>>>) -> Self {
            Self {
                penalty_graph: penalty_graph,
                vertices: vertices,
                edges: edges,
            }
        }

        pub fn initialize_with_epsilon(&mut self, epsilon: f64) {
        /*
        epsilon: precision with which vertices and edges exist
        Leave in the graph only those vertices and edges whoose
        penalties differ from minimum penalty in the group not more than by epsilon
        */
            self.initialize_vertices(epsilon);
            self.initialize_edges(epsilon);
        }

        pub fn initialize_vertices(&mut self, epsilon: f64) {
        /*
        epsilon: precision with which vertices exist
        (vertex = 1 if its penalty differs from minimum not more that by epsilon, else: vertex = 0)
        Fills self.vertices with true or false according to this rule
        */
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    let min_penalty_vertex = (self.penalty_graph.min_penalty_vertex(i, j)).1;
                    for d in 0..self.penalty_graph.max_disparity {
                        if j >= d
                        && self.penalty_graph.vertex_penalty_with_potentials(i, j, d) <=
                            min_penalty_vertex + epsilon {
                            self.vertices[i][j][d] = true;
                        } else {
                            self.vertices[i][j][d] = false;
                        }
                    }
                }
            }
        }

        pub fn initialize_edges(&mut self, epsilon: f64) {
        /*
        epsilon: precision with what edges exist
        (edge = 1 if its penalty differs from minimum not more that by epsilon, else: edge = 0)
        Fills self.edges with true or false according to this rule
        Go through each pair of neighbors.
        If edge weight of the pair is between
        min_penalty_edge and min_penalty_edge + epsilon that this edge exists.
        Else this edge doesn't exist
        */
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    for d in 0..self.penalty_graph.max_disparity {
                        for n in 0..4 {
                            if neighbor_exists(i, j, n,
                                               self.penalty_graph.left_image.len(),
                                               self.penalty_graph.left_image[0].len()) {
                                let min_penalty_edge =
                                    self.penalty_graph.min_penalty_edge(i, j, n);
                                for n_d in 0..self.penalty_graph.max_disparity {
                                    if self.penalty_graph.edge_exists(i, j, n, d, n_d)
                                    && self.penalty_graph.edge_penalty_with_potential(i, j, n, d, n_d)
                                        <= min_penalty_edge + epsilon {
                                        self.edges[i][j][d][n][n_d] = true;
                                    } else {
                                        self.edges[i][j][d][n][n_d] = false;
                                    }
                                }
                            } else {
                                for n_d in 0..self.penalty_graph.max_disparity {
                                    self.edges[i][j][d][n][n_d] = false;
                                }
                            }
                        }
                    }
                }
            }
        }

        pub fn crossing_out(&mut self) {
        /*
        Crosses out (makes `false`) a vertex, if an edge, that contains the vertex, is `false`.
        Crosses out an edge, if a vertex, that is connected with the edge, is `false`.
        Crossing out is finished when nothing was changed during one iteration

        */
            let mut change_indicator = true;
            while change_indicator {
                change_indicator = false;
                for i in 0..self.penalty_graph.left_image.len() {
                    for j in 0..self.penalty_graph.left_image[0].len() {
                        for d in 0..self.penalty_graph.max_disparity {
                            if j >= d {
                                if !self.vertices[i][j][d] {
                                    for n in 0..4 {
                                        if neighbor_exists(i, j, n,
                                                           self.penalty_graph.left_image.len(),
                                                           self.penalty_graph.left_image[0].len()) {
                                            let (n_i, n_j, n_index) = neighbor_index(i, j, n);
                                            for n_d in 0..self.penalty_graph.max_disparity {
                                                if self.penalty_graph.edge_exists(i, j, n, d, n_d)
                                                && self.edges[i][j][d][n][n_d] {
                                                    self.edges[i][j][d][n][n_d] = false;
                                                    self.edges[n_i][n_j][n_d][n_index][d] = false;
                                                    change_indicator = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                for n in 0..4 {
                                    if !neighbor_exists(i, j, n, self.penalty_graph.left_image.len(),
                                                        self.penalty_graph.left_image[0].len()) {
                                        continue;
                                    }
                                    let mut edge_exist = false;
                                    for n_d in 0..self.penalty_graph.max_disparity {
                                        if self.penalty_graph.edge_exists(i, j, n, d, n_d)
                                        && self.edges[i][j][d][n][n_d] {
                                            edge_exist = true;
                                            break;
                                        }
                                    }
                                    if !edge_exist && self.vertices[i][j][d] {
                                        self.vertices[i][j][d] = false;
                                        change_indicator = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        pub fn is_not_empty(&self) -> bool {
        /*
        crossing_out_graph.initialize_with_epsilon(epsilon);
        crossing_out_graph.crossing_out();ssible value of disparity
        Returns true if there are vertices and edges in a given graph
        */
            if self.vertices_exist() && self.edges_exist() {
                true
            } else {
                false
            }
        }

        pub fn vertices_exist(&self) -> bool {
        /*
        Returns true if there is at least one vertex in each object (pixel) of a graph
        */
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    let mut found: bool = false;
                    for d in 0..self.penalty_graph.max_disparity {
                        if j >= d && self.vertices[i][j][d] {
                            found = true;
                        }
                    }
                    if !found {
                        return false;
                    }
                }
            }
            true
        }

        pub fn edges_exist(&self) -> bool {
        /*
        Returns true if there is at least one edge between each pair of neighbors in a graph
        */
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    for n in 0..4 {
                        if !neighbor_exists(i, j, n,
                                            self.penalty_graph.left_image.len(),
                                            self.penalty_graph.left_image[0].len()) {
                            continue;
                        }
                        let mut found = false;
                        for d in 0..self.penalty_graph.max_disparity {
                            for n_d in 0..self.penalty_graph.max_disparity {
                                if !self.penalty_graph.edge_exists(i, j, n, d, n_d) {
                                    continue;
                                } else {
                                    if self.edges[i][j][d][n][n_d] {
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
            true
        }

        pub fn diffusion_while_not_consistent(&mut self, epsilon: f64, batch_size: usize) {
        /*
        epsilon: needed for initialization of crossing out graph
        batch_size: number of iterations after that crossing out will be done
        Calls diffusion while crossing out graph is empty after crossing out with given epsilon
        */
            let mut not_empty: bool = false;
            let mut i = 1;
            while !not_empty {
                self.penalty_graph.diffusion(i, batch_size);
                println!("{} iterations of diffusion ended", i + batch_size - 1);
                println!("Initializing crossing out graph with epsilon ...");
                self.initialize_with_epsilon(epsilon);
                if self.is_not_empty() {
                    println!("After initialization with epsilon, resulting graph is not empty");
                    println!("Crossing out ...");
                    self.crossing_out();
                    println!("Crossing out is done");
                    not_empty = self.is_not_empty();
                    if !not_empty {
                        println!("Need more diffusion inerations");
                    }
                }
                i += batch_size;
            }
            println!("Graph is consistent. Diffusion is done");
        }

        pub fn min_vertex_between_existing(&self, i: usize, j: usize) -> usize {
        /*
        (i, j): pixel coordinates
        Finds vertex with the minimum penalty among not-crossed out vertexes in pixel
        Returns disparity correspondent to this vertex
        */
            let mut min_vertex: f64 = f64::INFINITY;
            let mut disparity: usize = 0;
            let mut found: bool = false;
            for d in 0..self.penalty_graph.max_disparity {
                if j >= d && self.vertices[i][j][d] {
                    let current_vertex = self.penalty_graph.vertex_penalty_with_potentials(i, j, d);
                    if current_vertex < min_vertex {
                        min_vertex = current_vertex;
                        disparity = d;
                        found = true;
                    }
                }
            }
            assert!(found, "Disparity for pixel ({}, {}) wasn't found", i, j);
            disparity
        }

        pub fn cross_vertex(&mut self, i: usize, j: usize, disparity: usize) {
        /*
        (i, j): pixel coordinates
        Crosses out all vertexes (i, j, d) in pixel (i, j) except (i, j, disparity)
        (where d != given disparity)
        */
            assert!(self.vertices[i][j][disparity], "Choosen vertex [{}][{}][{}] is crossed out", i, j, disparity);
            for d in 0..self.penalty_graph.max_disparity {
                if self.vertices[i][j][d] && d != disparity {
                    self.vertices[i][j][d] = false;
                }
            }
        }

        pub fn find_best_labeling(&mut self) -> Vec<Vec<usize>> {
        /*
        Chooses the best vertex in the first pixel (0, 0) -> crosses out graph
        Do the same for all other pixels, ...
        */
            let mut disparity_map = vec![vec![0usize; self.penalty_graph.left_image[0].len()];
                                         self.penalty_graph.left_image.len()];
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    disparity_map[i][j] = self.min_vertex_between_existing(i, j);
                    assert!(disparity_map[i][j] <= j, "d > j for d = {}, j = {}, i = {}", disparity_map[i][j], j, i);
                    println!("Processed pixel ({}, {}) -> d = {}", i, j, disparity_map[i][j]);
                    self.cross_vertex(i, j, disparity_map[i][j]);
                    self.crossing_out();
                    assert!(self.is_not_empty(), "Graph is empty");
                }
            }
            disparity_map
        }

        pub fn check_disparity_map(&self, disparity_map: &Vec<Vec<usize>>) -> bool {
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..(self.penalty_graph.left_image[0].len() - 1) {
                    if j < disparity_map[i][j] || disparity_map[i][j + 1] > disparity_map[i][j] + 1 {
                        return false
                    }
                }
            }
            true
        }
    }

    #[test]
    fn test_min_penalty_zero() {
        let left_image = vec![vec![0u32; 2]; 1];
        let right_image = vec![vec![0u32; 2]; 1];
        let max_disparity = 2;
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity, 1.);
        let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
        let crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        let min_penalty_edge = crossing_out_graph.penalty_graph.min_penalty_edge(0, 0, 2);
        assert_eq!(0., min_penalty_edge);
        let min_penalty_vertex = (crossing_out_graph.penalty_graph.min_penalty_vertex(0, 0)).1;
        assert_eq!(0., min_penalty_vertex);
    }

    #[test]
    fn test_empty_crossing_out_graph() {
        let left_image = vec![vec![0u32; 2]; 1];
        let right_image = vec![vec![0u32; 2]; 1];
        let max_disparity = 1;
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity, 1.);
        let mut vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        vertices[0][0][0] = false;
        let mut edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
        edges[0][0][0][2][0] = true;
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        crossing_out_graph.crossing_out();
        println!("{:?}", crossing_out_graph.vertices);
        println!("{:?}", crossing_out_graph.edges);
        assert!(!crossing_out_graph.vertices_exist());
        assert!(!crossing_out_graph.edges_exist());
        assert!(!crossing_out_graph.is_not_empty());
    }

    #[test]
    fn test_crossing_out() {
        let left_image = [[1, 1].to_vec(), [1, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let max_disparity: usize = 2;
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity, 1.);
        let mut vertices = vec![vec![vec![true; max_disparity]; 2]; 2];
        let mut edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 2];
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        crossing_out_graph.edges[0][0][0][2][0] = false;
        crossing_out_graph.edges[0][1][0][0][0] = false;
        crossing_out_graph.edges[0][1][1][3][1] = false;
        crossing_out_graph.edges[1][1][1][1][1] = false;
        crossing_out_graph.crossing_out();
        assert!(crossing_out_graph.vertices[0][0][0]);
        assert!(crossing_out_graph.vertices[0][1][1]);
        assert!(crossing_out_graph.vertices[1][0][0]);
        assert!(crossing_out_graph.vertices[1][1][0]);
        assert!(!crossing_out_graph.vertices[0][1][0]);
        assert!(!crossing_out_graph.vertices[1][1][1]);
        assert!(crossing_out_graph.edges[0][0][0][2][1]);
        assert!(crossing_out_graph.edges[0][0][0][3][0]);
        assert!(crossing_out_graph.edges[1][0][0][2][0]);
        assert!(!crossing_out_graph.edges[1][0][0][2][1]);
        assert!(crossing_out_graph.edges[0][1][1][3][0]);
        assert!(!crossing_out_graph.edges[0][1][1][3][1]);
        assert!(!crossing_out_graph.edges[0][0][0][2][0]);
    }

    #[test]
    fn test_initialize_vertices() {
        let left_image = [[1, 1].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec()].to_vec();
        let max_disparity: usize = 2;
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.6;
        penalty_graph.potentials[0][1][0][0] = -13.7;
        penalty_graph.potentials[0][1][0][1] = 80.;
        let mut vertices = vec![vec![vec![false; max_disparity]; 2]; 1];
        let mut edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        crossing_out_graph.initialize_vertices(1.);
        assert!(crossing_out_graph.vertices[0][0][0]);
        assert!(!crossing_out_graph.vertices[0][1][0]);
        assert!(crossing_out_graph.vertices[0][1][1]);
    }

    #[test]
    fn test_initialize_edges() {
        let left_image = [[1, 1, 0].to_vec()].to_vec();
        let right_image = [[1, 0, 0].to_vec()].to_vec();
        let max_disparity: usize = 2;
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.6;
        penalty_graph.potentials[0][1][0][0] = -13.7;
        penalty_graph.potentials[0][1][0][1] = 80.;
        penalty_graph.potentials[0][1][2][0] = 358.;
        penalty_graph.potentials[0][1][2][1] = -1E9;
        penalty_graph.potentials[0][2][0][0] = -0.3;
        penalty_graph.potentials[0][2][0][1] = 0.1;
        let mut vertices = vec![vec![vec![false; max_disparity]; 3]; 1];
        let mut edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 3]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        crossing_out_graph.initialize_edges(0.1);
        assert!(crossing_out_graph.edges[0][0][0][2][0]);
        assert!(!crossing_out_graph.edges[0][0][0][2][1]);
        assert!(!crossing_out_graph.edges[0][1][0][2][0]);
        assert!(!crossing_out_graph.edges[0][1][0][2][1]);
        assert!(!crossing_out_graph.edges[0][1][1][2][0]);
        assert!(crossing_out_graph.edges[0][1][1][2][1]);
    }

    #[test]
    fn test_initialize_with_epsilon() {
        let left_image = [[1, 1].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec()].to_vec();
        let max_disparity: usize = 2;
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.6;
        penalty_graph.potentials[0][1][0][0] = -0.3;
        penalty_graph.potentials[0][1][0][1] = 0.1;
        let mut vertices = vec![vec![vec![false; max_disparity]; 2]; 1];
        let mut edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        crossing_out_graph.initialize_with_epsilon(1.3);
        assert!(crossing_out_graph.vertices[0][0][0]);
        assert!(!crossing_out_graph.vertices[0][1][0]);
        assert!(crossing_out_graph.vertices[0][1][1]);
        assert!(crossing_out_graph.edges[0][0][0][2][0]);
        assert!(!crossing_out_graph.edges[0][0][0][2][1]);
    }

    #[test]
    fn test_vertices_exist() {
        let left_image = [[1, 1].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec()].to_vec();
        let max_disparity: usize = 2;
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity, 1.);
        let mut vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let mut edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        crossing_out_graph.vertices[0][1][1] = false;
        assert!(crossing_out_graph.vertices_exist());
    }

    #[test]
    fn test_vertices_exist_negative() {
        let left_image = [[1, 1].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec()].to_vec();
        let max_disparity: usize = 1;
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity, 1.);
        let mut vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let mut edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        crossing_out_graph.vertices[0][0][0] = false;
        assert!(!crossing_out_graph.vertices_exist());
    }
}
