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
        penalty_graph: PenaltyGraph,
        // A vertice is defined by pixel coordinates (i, j) and disparity value d
        vertices: Vec<Vec<Vec<bool>>>,
        // And edge is defined by a vertex (i, j, d),
        // a number of its neighbour n (from 0 to 3) and disparity n_d of the neighbour
        edges: Vec<Vec<Vec<Vec<Vec<bool>>>>>
    }

    impl CrossingOutGraph {
        pub fn initialize(&mut self, max_disparity: usize, epsilon: f64) {
            self.initialize_vertices(max_disparity, epsilon);
            self.initialize_edges(max_disparity, epsilon);
        }

        pub fn initialize_vertices(&mut self, max_disparity: usize, epsilon: f64) {
        /*
        max_disparity: maximul possible value of disparity
        epsilon: precision with what vertices exist
        (vertex = 1 if its penalty differs from minimum not more that by epsilon, else: vertex = 0)
        Fills self.vertices with true or false according to this rule
        */
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    let min_penalty_vertex = self.min_penalty_vertex(i, j, max_disparity);
                    for d in 0..max_disparity {
                        if j >= d
                        && self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                            as usize][self.penalty_graph.right_image[i][j-d] as usize]
                            + self.sum_of_potentials(i, j, d)
                            >= min_penalty_vertex
                        && self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                            as usize][self.penalty_graph.right_image[i][j-d] as usize]
                            + self.sum_of_potentials(i, j, d)
                            <= min_penalty_vertex + epsilon {
                                self.vertices[i][j][d] = true;
                        } else {
                            self.vertices[i][j][d] = false;
                        }
                    }
                }
            }
        }

        pub fn initialize_edges(&mut self, max_disparity: usize, epsilon: f64) {
        /*
        max_disparity: maximul possible value of disparity
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
                    for d in 0..max_disparity {
                        if j >= d {
                            for n in 0..4 {
                                if neighbour_exists(i as i32, j as i32, n,
                                                    self.penalty_graph.left_image.len() as i32,
                                                    self.penalty_graph.left_image[0].len() as i32) {
                                    let (n_i, n_j, n_index) = neighbour_index(i, j, n);
                                    let min_penalty_edge =
                                    self.min_penalty_edge(max_disparity, i, j, n, n_i, n_j, n_index);
                                    for n_d in 0..max_disparity {
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
                                    for n_d in 0..max_disparity {
                                        self.edges[i][j][d][n][n_d] = false;
                                    }
                                }
                            }
                        } else {
                            for n in 0..4 {
                                for n_d in 0..max_disparity {
                                    self.edges[i][j][d][n][n_d] = false;
                                }
                            }
                        }
                    }
                }
            }
        }

        pub fn min_penalty_vertex(&mut self, i: usize, j: usize, max_disparity: usize) -> f64 {
        /*
        i: number of pixel row in image
        j: number of pixel column in image
        (i, j) defines pixel (its coordinate in image)
        max_disparity: maximum possible disparity value
        Returns minimum penalty of given pixel (updated by potentials):
        look over each possible diparity value of pixel and choose minimum value of vertex penalty
        */
            let mut min_penalty_vertex: f64 = f64::INFINITY;
            for d in 0..max_disparity {
                if j >= d && min_penalty_vertex > self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                as usize][self.penalty_graph.right_image[i][j - d] as usize]
                + self.sum_of_potentials(i, j, d) {
                    min_penalty_vertex = self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                        as usize][self.penalty_graph.right_image[i][j - d] as usize]
                        + self.sum_of_potentials(i, j, d);
                }
            }
            min_penalty_vertex
        }

        pub fn sum_of_potentials(&mut self, i: usize, j: usize, d: usize) -> f64 {
        /*
        max_disparity: maximum possible disparity value
        i: number of pixel row in image
        j: number of pixel column in image
        d: disparity of pixel (i, j)
        Returns the sum of potentials between pixel (i, j) with disparity d and all its neighbours
        */
            let mut sum = 0.;
            for n in 0..4 {
                if neighbour_exists(i as i32, j as i32, n,
                                    self.penalty_graph.left_image.len() as i32,
                                    self.penalty_graph.left_image[0].len() as i32) {
                    sum += self.penalty_graph.potentials[i][j][n][d];
                }
            }
            sum
        }

        pub fn min_penalty_edge(&mut self, max_disparity: usize, i: usize, j: usize, n: usize, n_i: usize, n_j: usize, n_index: usize) -> f64 {
        /*
        max_disparity: maximum disparity value that is possible
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
                for d in 0..max_disparity {
                    for n_d in 0..max_disparity {
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

        pub fn crossing_out(&mut self, max_disparity: usize) {
            let mut change_indicator = true;
            while change_indicator {
                change_indicator = false;
                for i in 0..self.penalty_graph.left_image.len() {
                    for j in 0..self.penalty_graph.left_image[0].len() {
                        for d in 0..max_disparity {
                            if !self.vertices[i][j][d] {
                                for n in 0..3 {
                                    if neighbour_exists(i as i32, j as i32, n,
                                                        self.penalty_graph.left_image.len() as i32,
                                                        self.penalty_graph.left_image[0].len() as i32) {
                                        for n_d in 0..max_disparity {
                                            if self.edges[i][j][d][n][n_d] {
                                                self.edges[i][j][d][n][n_d] = false;
                                                change_indicator = true;
                                            }
                                        }
                                    }
                                }
                            }
                            for n in 0..3 {
                                if neighbour_exists(i as i32, j as i32, n,
                                                    self.penalty_graph.left_image.len() as i32,
                                                    self.penalty_graph.left_image[0].len() as i32) {
                                    let mut edge_exist = false;
                                    for n_d in 0..max_disparity {
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
    }

    #[test]
    fn test_min_penalty_zero() {
        let left_image = vec![vec![0u32; 2]; 1];
        let right_image = vec![vec![0u32; 2]; 1];
        let max_disparity = 2;
        let mut penalty_graph = PenaltyGraph {lookup_table : vec![vec![0.; 255]; 255],
                                              potentials : vec![vec![vec![vec![0f64; max_disparity]; 4]; 2]; 1],
                                              left_image : vec![vec![0u32; 2]; 1],
                                              right_image : vec![vec![0u32; 2]; 1]};
        penalty_graph.initialize(left_image, right_image, max_disparity);
        let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph {penalty_graph : penalty_graph,
                                                       vertices: vertices,
                                                       edges : edges};
        let min_penalty_edge = crossing_out_graph.min_penalty_edge(max_disparity, 0, 0, 2, 0, 1, 0);
        assert_eq!(0., min_penalty_edge);
        let min_penalty_vertex = crossing_out_graph.min_penalty_vertex(0, 0, max_disparity);
        assert_eq!(0., min_penalty_vertex);
    }

    #[test]
    fn test_min_penalty() {
        let left_image = [[5, 1].to_vec()].to_vec();
        let right_image = [[2, 4].to_vec()].to_vec();
        let max_disparity = 2;
        let mut penalty_graph = PenaltyGraph {lookup_table : vec![vec![0.; 255]; 255],
                                              potentials : vec![vec![vec![vec![0f64; max_disparity]; 4]; 2]; 1],
                                              left_image : vec![vec![0u32; 2]; 1],
                                              right_image : vec![vec![0u32; 2]; 1]};
        penalty_graph.initialize(left_image, right_image, max_disparity);
        let vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        let edges = vec![vec![vec![vec![vec![true; max_disparity]; 4]; max_disparity]; 2]; 1];
        let mut crossing_out_graph = CrossingOutGraph {penalty_graph : penalty_graph,
                                                       vertices: vertices,
                                                       edges : edges};
        let min_penalty_edge = crossing_out_graph.min_penalty_edge(max_disparity, 0, 0, 2, 0, 1, 0);
        assert_eq!(0., min_penalty_edge);
        assert_eq!(3., crossing_out_graph.min_penalty_vertex(0, 0, max_disparity));
        assert_eq!(1., crossing_out_graph.min_penalty_vertex(0, 1, max_disparity));
        let mut vertices = vec![vec![vec![true; max_disparity]; 2]; 1];
        vertices[0][0][1] = false;
        vertices[0][1][0] = false;
        crossing_out_graph.initialize(max_disparity, 1.);
        assert_eq!(vertices, crossing_out_graph.vertices);
        let mut edges = vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; 2]; 1];
        edges[0][0][0][2][0] = true;
        edges[0][0][0][2][1] = true;
        edges[0][1][0][0][0] = true;
        edges[0][1][0][0][1] = true;
        edges[0][1][1][0][0] = true;
        edges[0][1][1][0][1] = true;
        crossing_out_graph.crossing_out(max_disparity);
        assert_eq!(vertices, crossing_out_graph.vertices);
        edges[0][1][0][0][0] = false;
        edges[0][1][0][0][1] = false;
        assert_eq!(edges, crossing_out_graph.edges);
    }
}
