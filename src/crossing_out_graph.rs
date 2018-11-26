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
    use super::super::diffusion::diffusion::neighbour_exists;
    use super::super::diffusion::diffusion::neighbour_index;

    #[derive(Debug)]
    pub struct CrossingOutGraph {
        pub penalty_graph: PenaltyGraph,
        // A vertice is defined by pixel coordinates (i, j) and disparity value d
        vertices: Vec<Vec<Vec<bool>>>,
        // And edge is defined by a vertex (i, j, d),
        // a number of its neighbour n (from 0 to 3) and disparity n_d of the neighbour
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
                    let min_penalty_vertex = self.min_penalty_vertex(i, j);
                    for d in 0..self.penalty_graph.max_disparity {
                        if j >= d
                        && self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                            as usize][self.penalty_graph.right_image[i][j-d] as usize]
                            + self.penalty_graph.sum_of_potentials(i, j, d)
                            >= min_penalty_vertex
                        && self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                            as usize][self.penalty_graph.right_image[i][j-d] as usize]
                            + self.penalty_graph.sum_of_potentials(i, j, d)
                            <= min_penalty_vertex + epsilon {
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
        Go through each pair of neighbours.
        If edge weight of the pair is between
        min_penalty_edge and min_penalty_edge + epsilon that this edge exists.
        Else this edge doesn't exist
        */
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    for d in 0..self.penalty_graph.max_disparity {
                        for n in 0..4 {
                            if neighbour_exists(i, j, n,
                                                self.penalty_graph.left_image.len(),
                                                self.penalty_graph.left_image[0].len()) {
                                let (n_i, n_j, n_index) = neighbour_index(i, j, n);
                                let min_penalty_edge =
                                self.min_penalty_edge(i, j, n, n_i, n_j, n_index);
                                for n_d in 0..self.penalty_graph.max_disparity {
                                    if self.penalty_graph.lookup_table[d][n_d]
                                    - self.penalty_graph.potentials[i][j][n][d]
                                    - self.penalty_graph.potentials[n_i][n_j][n_index][n_d]
                                    >= min_penalty_edge
                                    && self.penalty_graph.lookup_table[d][n_d]
                                    - self.penalty_graph.potentials[i][j][n][d]
                                    - self.penalty_graph.potentials[n_i][n_j][n_index][n_d]
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

        pub fn min_penalty_vertex(&mut self, i: usize, j: usize) -> f64 {
        /*
        i: number of pixel row in image
        j: number of pixel column in image
        (i, j) defines pixel (its coordinate in image)
        Returns minimum penalty of given pixel (updated by potentials):
        look over each possible diparity value of pixel and choose minimum value of vertex penalty
        */
            let mut min_penalty_vertex: f64 = f64::INFINITY;
            for d in 0..self.penalty_graph.max_disparity {
                if j >= d && min_penalty_vertex > self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                as usize][self.penalty_graph.right_image[i][j - d] as usize]
                + self.penalty_graph.sum_of_potentials(i, j, d) {
                    min_penalty_vertex = self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                        as usize][self.penalty_graph.right_image[i][j - d] as usize]
                        + self.penalty_graph.sum_of_potentials(i, j, d);
                }
            }
            min_penalty_vertex
        }

        pub fn min_penalty_edge(&mut self, i: usize, j: usize, n: usize,
                                n_i: usize, n_j: usize, n_index: usize) -> f64 {
        /*
        i: row of pixel in left image
        j: column of pixel in left image
        n_i: row of pixel neighbour in left image
        n_j: column of pixel neighbour in left image
        n_index: index of pixel for neighbour (from 0 to 3)
        returns min_{k'} g*_{tt'}(d, d'), where t is pixel (i, j), t' is it neighbour,
        g*_{tt'}(d, d') = g_{tt'}(d, d') - phi_{tt'}(k) - phi_{t't}(k),
        where phi are potentials
        So, we have fixed pixel and its neighbour;
        and search for minimum edge penalty (with potentials) between them
        based on pixel disparity and neighbour disparity
        */
            let mut min_penalty_edge: f64 = f64::INFINITY;
                for d in 0..self.penalty_graph.max_disparity {
                    for n_d in 0..self.penalty_graph.max_disparity {
                        if min_penalty_edge > self.penalty_graph.lookup_table[d][n_d]
                        - self.penalty_graph.potentials[i][j][n][d]
                        - self.penalty_graph.potentials[n_i][n_j][n_index][n_d] {
                            min_penalty_edge = self.penalty_graph.lookup_table[d][n_d]
                            - self.penalty_graph.potentials[i][j][n][d]
                            - self.penalty_graph.potentials[n_i][n_j][n_index][n_d];
                        }
                    }
                }
            min_penalty_edge
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
                            if !self.vertices[i][j][d] {
                                for n in 0..3 {
                                    if neighbour_exists(i, j, n,
                                                        self.penalty_graph.left_image.len(),
                                                        self.penalty_graph.left_image[0].len()) {
                                        for n_d in 0..self.penalty_graph.max_disparity {
                                            if self.edges[i][j][d][n][n_d] {
                                                self.edges[i][j][d][n][n_d] = false;
                                                change_indicator = true;
                                            }
                                        }
                                    }
                                }
                            }
                            for n in 0..3 {
                                if neighbour_exists(i, j, n,
                                                    self.penalty_graph.left_image.len(),
                                                    self.penalty_graph.left_image[0].len()) {
                                    let mut edge_exist = false;
                                    for n_d in 0..self.penalty_graph.max_disparity {
                                        if self.edges[i][j][d][n][n_d] {
                                            edge_exist = true;
                                        }
                                    }
                                    if !edge_exist && self.vertices[i][j][d] {
                                        self.vertices[i][j][d] = false;
                                        change_indicator = true;
                                        continue;
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
        Returns true if there is at least one vertex in a graph
        */
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    for d in 0..self.penalty_graph.max_disparity {
                        if self.vertices[i][j][d] {
                            return true;
                        }
                    }
                }
            }
            false
        }

        pub fn edges_exist(&self) -> bool {
        /*
        Returns true if there is at least one edge in a graph
        */
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    for d in 0..self.penalty_graph.max_disparity {
                        for n in 0..4 {
                            for n_d in 0..self.penalty_graph.max_disparity {
                                if self.edges[i][j][d][n][n_d] {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            false
        }
    }

    #[test]
    fn test_min_penalty_zero() {
        let left_image = vec![vec![0u32; 2]; 1];
        let right_image = vec![vec![0u32; 2]; 1];
        let max_disparity = 2;
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity);
        let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        let min_penalty_edge = crossing_out_graph.min_penalty_edge(0, 0, 2, 0, 1, 0);
        assert_eq!(0., min_penalty_edge);
        let min_penalty_vertex = crossing_out_graph.min_penalty_vertex(0, 0);
        assert_eq!(0., min_penalty_vertex);
    }

    #[test]
    fn test_min_penalty() {
        let left_image = [[5, 1].to_vec()].to_vec();
        let right_image = [[2, 4].to_vec()].to_vec();
        let max_disparity = 2;
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity);
        let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph::initialize(penalty_graph, vertices, edges);
        let min_penalty_edge = crossing_out_graph.min_penalty_edge(0, 0, 2, 0, 1, 0);
        assert_eq!(0., min_penalty_edge);
        assert_eq!(3., crossing_out_graph.min_penalty_vertex(0, 0));
        assert_eq!(1., crossing_out_graph.min_penalty_vertex(0, 1));
        let mut vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        vertices[0][0][1] = false;
        vertices[0][1][0] = false;
        crossing_out_graph.initialize_with_epsilon(1.);
        assert_eq!(vertices, crossing_out_graph.vertices);
        let mut edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
        edges[0][0][0][2][0] = true;
        edges[0][0][0][2][1] = true;
        edges[0][1][0][0][0] = true;
        edges[0][1][0][0][1] = true;
        edges[0][1][1][0][0] = true;
        edges[0][1][1][0][1] = true;
        crossing_out_graph.crossing_out();
        assert_eq!(vertices, crossing_out_graph.vertices);
        edges[0][1][0][0][0] = false;
        edges[0][1][0][0][1] = false;
        assert_eq!(edges, crossing_out_graph.edges);
        assert!(true, crossing_out_graph.vertices_exist());
        assert!(true, crossing_out_graph.edges_exist());
        assert!(true, crossing_out_graph.is_not_empty());
    }

    #[test]
    fn test_empty_crossing_out_graph() {
        let left_image = vec![vec![0u32; 2]; 1];
        let right_image = vec![vec![0u32; 2]; 1];
        let max_disparity = 1;
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, max_disparity);
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
}
