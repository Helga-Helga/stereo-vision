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
pub mod penalty_graph {
    use std::f64;
    use super::super::diffusion::diffusion::neighbour_exists;
    use super::super::diffusion::diffusion::neighbour_index;
    use super::super::diffusion::diffusion::number_of_neighbours;

    #[derive(Debug)]
    pub struct PenaltyGraph {
     pub lookup_table: Vec<Vec<f64>>,
     pub potentials: Vec<Vec<Vec<Vec<f64>>>>,
     pub left_image: Vec<Vec<u32>>,
     pub right_image: Vec<Vec<u32>>,
     pub max_disparity: usize,
    }

    impl PenaltyGraph {
        pub fn initialize(left_image: Vec<Vec<u32>>,
                          right_image: Vec<Vec<u32>>,
                          max_disparity: usize) -> Self {
            assert_eq!(left_image.len(), right_image.len());
            assert_eq!(left_image[0].len(), right_image[0].len());
            let mut lookup_table: Vec<Vec<f64>> = vec![vec![0f64; 256]; 256];
            for i in 0..256 {
                for j in 0..256 {
                    lookup_table[i][j] = (i as i32 - j as i32).abs() as f64;
                }
            }
            Self {
                lookup_table: lookup_table,
                potentials: vec![vec![vec![vec![0f64; max_disparity]; 4]; left_image[0].len()]; left_image.len()],
                left_image: left_image,
                right_image: right_image,
                max_disparity: max_disparity,
            }
        }

        pub fn vertex_penalty(&self, left_intensity: usize, right_intensity: usize) -> f64 {
        /*
        left_intensity: intensity of a pixel in the left image
        right_intensity: intensity of a pixel in the right image
        Returns an absolute value of left and right intensity defference
        */
            self.lookup_table[left_intensity][right_intensity] as f64
        }

        pub fn edge_penalty(&self, disparity: usize, disparity_neighbour: usize) -> f64 {
        /*
        disparity: disparity of a pixel (a shift of a pixel between left and right images)
        disparity_neighbour: disparity of a neighbour
        Returns an absolute value of pixel dispatity and its neighbour disparity
        */
            self.lookup_table[disparity][disparity_neighbour] as f64
        }

        pub fn penalty(&self, disparity_map: Vec<Vec<usize>>) -> f64 {
        /*
        disparity_map: matrix of the same size as an image,
        in cell (i, j) contains disparity of pixel (i, j)
        Returns a general penalty: sum_t f_t(k_t) + sum_{tt' in tau} g_{tt'}(k_t, k_t'), where
        f_t(k_t) is a penalty of vertex in pixel t with disparity k_t,
        tau is a set of neighbours, if tt' in tau, then they are neighbours,
        g_{tt'}(k_t, k_t') is a penalty of edge that connects
        a vertice in pixel t with disparity k_t and a neighbour pixel t' with disparity k_t'
        */
            let mut penalty: f64 = 0.;
            for i in 0..self.left_image.len() {
                for j in 0..self.left_image[0].len() {
                    if j >= disparity_map[i][j] as usize {
                        penalty +=
                        self.vertex_penalty(self.left_image[i][j] as usize,
                                            self.right_image[i][j - disparity_map[i][j]] as usize) +
                        self.sum_of_potentials(i, j, disparity_map[i][j]);
                    } else {
                        penalty += f64::INFINITY;
                    }
                    for n in 0..4 {
                        if neighbour_exists(i, j, n, self.left_image.len(), self.left_image[0].len()) {
                            let (n_i, n_j, n_index) = neighbour_index(i, j, n);
                            penalty +=
                                self.edge_penalty(disparity_map[i][j], disparity_map[n_i][n_j]) -
                                self.potentials[i][j][n][disparity_map[i][j]] -
                                self.potentials[n_i][n_j][n_index][disparity_map[n_i][n_j]];
                        }
                    }
                }
            }
            penalty
        }

        pub fn sum_of_potentials(&self, i: usize, j: usize, d: usize) -> f64 {
        /*
        i: number of pixel row in image
        j: number of pixel column in image
        d: disparity of pixel (i, j)
        Returns the sum of potentials between pixel (i, j) with disparity d and all its neighbours
        */
            let mut sum = 0.;
            for n in 0..4 {
                if neighbour_exists(i, j, n,
                                    self.left_image.len(),
                                    self.left_image[0].len()) {
                    sum += self.potentials[i][j][n][d];
                }
            }
            sum
        }

        pub fn min_edge_between_neighbours(&self, i: usize, j: usize, n: usize, d: usize) -> f64 {
        /*
        (i, j): coordinate of a pixel in an image
        n: number of a neighbour (from 0 to 3)
        d: disparity of pixel (i, j)
        Returns min_{n_d} g_{tt'}(d, n_d), where t = (i, j), t' is a neighbour of t,
        n_d is a disparity in pixel t'
        */
            let mut result: f64 = 0.;
            let (n_i, n_j, n_index) = neighbour_index(i, j, n);
            for n_d in 0..self.max_disparity {
                if result > self.lookup_table[d][n_d] - self.potentials[i][j][n][d] -
                            self.potentials[n_i][n_j][n_index][n_d] {
                    result =
                        self.lookup_table[d][n_d] -
                        self.potentials[i][j][n][d] -
                        self.potentials[n_i][n_j][n_index][n_d];
                }
            }
            result
        }

        pub fn sum_min_edges(&self, i: usize, j: usize, d: usize) -> f64 {
        /*
        (i, j): coordnate of a pixel in an image
        d: disparity in pixel (i, j)
        Used equation: sum_{t' in N(t)} min_{n_d} g_{tt'}(d, n_d),
        where t = (i, j), t' is a neighbour of t, N(t) is a set of neighbours of pixel t,
        n_d is disparity in pixel t'.
        Returns the sum of minimum edges from pixel to all its neighbours
        */
            let mut result: f64 = 0.;
            for n in 0..4 {
                if neighbour_exists(i, j, n, self.left_image.len(), self.left_image[0].len()) {
                    result += self.min_edge_between_neighbours(i, j, n, d);
                }
            }
            result
        }

        pub fn update_potential(&mut self, i: usize, j: usize, d: usize, n: usize){
        /*
        (i, j): coordinate of pixel t in an image
        d: disparity in pixel t
        n: number of pixel neighbour (from 0 to 3)
        Returns phi_{tt'}(k_t) = min_{k_t'} g_{tt'}(k_t, k_t') -
        - [f_t(k_t) + sum_{t' in N(t)} min_{k_t'} g_{tt'}(k_t, k_t')] / [|N(t)| + 1], where
        t' is n`th neighbour of pixel t,
        N(t) is a set of neighbours of pixel t
        f_t(k_t) is a penalty of vertex in pixel t with disparity k_t
        g_{tt'}(k_t, k_t') is a penalty of an edge between
        a vertex in pixel t with penalty k_t and a vertex in neighbour pixel t' with penalty k_t'
        */
        let vertex_penalty: f64 =
            self.vertex_penalty(self.left_image[i][j] as usize,
            self.right_image[i][j - d] as usize);
        let number_of_neighbours: f64 =
            number_of_neighbours(i, j, n, self.left_image.len(), self.left_image[0].len()) as f64;
        self.potentials[i][j][n][d] = self.min_edge_between_neighbours(i, j, n, d) -
            (vertex_penalty + self.sum_min_edges(i, j, d)) / (number_of_neighbours + 1.)
        }
    }

    #[test]
    fn test_penalty_one_pixel() {
        let left_image = vec![vec![0u32; 1]; 1];
        let right_image = vec![vec![0u32; 1]; 1];
        let disparity_map = vec![vec![0usize; 1]; 1];
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, 1);
        println!("{:?}", penalty_graph.potentials);
        assert_eq!(0., penalty_graph.penalty(disparity_map));
    }

    #[test]
    fn test_penalty_four_pixels_inf() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let disparity_map = vec![vec![1usize; 2]; 2];
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2);
        assert_eq!(f64::INFINITY, penalty_graph.penalty(disparity_map));
    }

    #[test]
    fn test_penalty_four_pixels() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let disparity_map = [[0, 1].to_vec(), [0, 1].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2);
        assert_eq!(4., penalty_graph.penalty(disparity_map));
        penalty_graph.potentials[0][0][2][0] = 1.;
        let new_disparity_map = [[0, 1].to_vec(), [0, 1].to_vec()].to_vec();
        assert_eq!(3., penalty_graph.penalty(new_disparity_map));
    }
 }
