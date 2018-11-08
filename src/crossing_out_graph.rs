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
    use super::super::penalty_graph::penalty_graph::PenaltyGraph;
    use super::super::diffusion::diffusion::neighbour_exists;
    use super::super::diffusion::diffusion::neighbour_index;
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
            let min_penalty_vertex = self.min_penalty_vertex(max_disparity);
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    for d in 0..max_disparity {
                        if self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                            as usize][self.penalty_graph.right_image[i][j-d] as usize]
                            + self.sum_of_potentials(max_disparity, i, j, d)
                            >= min_penalty_vertex
                        && self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                            as usize][self.penalty_graph.right_image[i][j-d] as usize]
                            + self.sum_of_potentials(max_disparity, i, j, d)
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
        */
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    for d in 0..max_disparity {
                        for n in 0..4 {
                            let min_penalty_edge = self.min_penalty_edge(max_disparity, i, j, d, n);
                            for n_d in 0..max_disparity {
                                if neighbour_exists(i as i32, j as i32, n,
                                                    self.penalty_graph.left_image.len() as i32,
                                                    self.penalty_graph.left_image[0].len() as i32) {
                                    let (n_i, n_j, n_index) = neighbour_index(i, j, n);
                                    if self.penalty_graph.lookup_table[d][n_d]
                                    - self.penalty_graph.potentials[i][j][d][n]
                                    - self.penalty_graph.potentials[n_i][n_j][n_d][n_index]
                                    >= min_penalty_edge
                                    && self.penalty_graph.lookup_table[d][n_d]
                                    - self.penalty_graph.potentials[i][j][d][n]
                                    - self.penalty_graph.potentials[n_i][n_j][n_d][n_index]
                                    <= min_penalty_edge + epsilon {
                                        self.edges[i][j][d][n][n_d] = true;
                                    } else {
                                        self.edges[i][j][d][n][n_d] = false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        pub fn min_penalty_vertex(&self, max_disparity: usize) -> f64 {
        /*
        max_disparity: maximum possible disparity value
        Returns minimum penalty of vertices (updated by potentials)
        */
            let mut min_penalty_vertex: f64 = 0.;
            for i in 0..self.penalty_graph.left_image.len() {
                for j in 0..self.penalty_graph.left_image[0].len() {
                    for d in 0..max_disparity {
                        if self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                            as usize][self.penalty_graph.right_image[i][j-d] as usize]
                            + self.sum_of_potentials(max_disparity, i, j, d)
                            < min_penalty_vertex {
                            min_penalty_vertex =
                                self.penalty_graph.lookup_table[self.penalty_graph.left_image[i][j]
                                as usize][self.penalty_graph.right_image[i][j-d] as usize]
                                + self.sum_of_potentials(max_disparity, i, j, d);
                        }
                    }
                }
            }
            min_penalty_vertex
        }

        pub fn sum_of_potentials(&self, max_disparity: usize, i: usize, j: usize, d: usize) -> f64 {
        /*
        max_disparity: maximum possible disparity value
        i: number of pixel row in image
        j: number of pixel column in image
        d: disparity of pixel (i, j)
        Returns the sum of potentials between pixel (i, j) with disparity d and all its neighbours
        */
            let mut sum = 0.;
            for n in 0..max_disparity {
                if neighbour_exists(i as i32, j as i32, n,
                                    self.penalty_graph.left_image.len() as i32,
                                    self.penalty_graph.left_image[0].len() as i32) {
                    sum += self.penalty_graph.potentials[i][j][d][n];
                }
            }
            sum
        }

        pub fn min_penalty_edge(&self, max_disparity: usize, i: usize, j: usize, d: usize, n: usize) -> f64 {
        /*
        max_disparity: maximum disparity value that is possible
        i: row of pixel in left image
        j: column of pixel in left image
        d: disparity of pixel (i, j)
        n: number of pixel neighbour (from 0 to 3)
        n_d: disparity of neighbour
        returns min_{k'} g*_{tt'}(d, d'), where t is pixel (i, j), t' is it neighbour,
        g*_{tt'}(d, d') = g_{tt'}(d, d') - phi_{tt'}(k) - phi_{t't}(k),
        where phi are potentials
        So, we have fixel pixel, its disparity and neighbour;
        and search for minimum edge penalty (with potentials) between them based on neighbour disparity
        */
            let mut min_penalty_edge: f64 = 0.;
            if neighbour_exists(i as i32, j as i32, n,
                                self.penalty_graph.left_image.len() as i32,
                                self.penalty_graph.left_image[0].len() as i32) {
                for n_d in 0..max_disparity {
                    let (n_i, n_j, n_index) = neighbour_index(i, j, n);
                    if self.penalty_graph.lookup_table[d][n_d]
                    - self.penalty_graph.potentials[i][j][d][n]
                    - self.penalty_graph.potentials[n_i][n_j][n_d][n_index]
                    < min_penalty_edge {
                        min_penalty_edge = self.penalty_graph.lookup_table[d][n_d]
                        - self.penalty_graph.potentials[i][j][d][n]
                        - self.penalty_graph.potentials[n_i][n_j][n_d][n_index];
                    }
                }
            }
            min_penalty_edge

        }
    }
}
