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

    #[derive(Debug)]
    pub struct PenaltyGraph {
     pub lookup_table: Vec<Vec<f64>>,
     pub potentials: Vec<Vec<Vec<Vec<f64>>>>,
     pub left_image: Vec<Vec<u32>>,
     pub right_image: Vec<Vec<u32>>
    }

    impl PenaltyGraph {
        pub fn initialize(&mut self,
                       left_image: Vec<Vec<u32>>,
                       right_image: Vec<Vec<u32>>,
                       max_disparity: usize) {
            assert_eq!(left_image.len(), right_image.len());
            assert_eq!(left_image[0].len(), right_image[0].len());
            for i in 0..256 {
                for j in 0..256 {
                    self.lookup_table[i][j] = (i as i32 - j as i32).abs() as f64;
                }
            }
            self.potentials =
                vec![vec![vec![vec![0f64; max_disparity]; 4]; left_image[0].len()]; left_image.len()];
            self.left_image = left_image;
            self.right_image = right_image;
        }

        pub fn vertex_penalty(&self, left_intensity: usize, right_intensity: usize) -> f64 {
            self.lookup_table[left_intensity][right_intensity] as f64
        }

        pub fn edge_penalty(&self, disparity: usize, disparity_neighbour: usize) -> f64 {
            self.lookup_table[disparity][disparity_neighbour] as f64
        }

        pub fn penalty(&self, disparity_map: Vec<Vec<usize>>) -> f64 {
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
        max_disparity: maximum possible disparity value
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
    }

    #[test]
    fn test_penalty_one_pixel() {
        let left_image = vec![vec![0u32; 1]; 1];
        let right_image = vec![vec![0u32; 1]; 1];
        let disparity_map = vec![vec![0usize; 1]; 1];
        let mut penalty_graph = PenaltyGraph {lookup_table : vec![vec![0.; 256]; 256],
                                              potentials : vec![vec![vec![vec![0f64; 1]; 4]; 1]; 1],
                                              left_image : vec![vec![0; 1]; 1],
                                              right_image : vec![vec![0; 1]; 1]};
        penalty_graph.initialize(left_image, right_image, 1);
        assert_eq!(0., penalty_graph.penalty(disparity_map));
    }

    #[test]
    fn test_penalty_four_pixels() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let disparity_map = vec![vec![1usize; 2]; 2];
        let mut penalty_graph = PenaltyGraph {lookup_table : vec![vec![0.; 256]; 256],
                                              potentials : vec![vec![vec![vec![0f64; 2]; 4]; 2]; 2],
                                              left_image : vec![vec![0; 2]; 2],
                                              right_image : vec![vec![0; 2]; 2]};
        penalty_graph.initialize(left_image, right_image, 2);
        assert_eq!(f64::INFINITY, penalty_graph.penalty(disparity_map));
    }

    #[test]
    fn test_penalty_four_pixels_zero() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let disparity_map = [[0, 1].to_vec(), [0, 1].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph {lookup_table : vec![vec![0.; 256]; 256],
                                              potentials : vec![vec![vec![vec![0f64; 2]; 4]; 2]; 2],
                                              left_image : vec![vec![0; 2]; 2],
                                              right_image : vec![vec![0; 2]; 2]};
        penalty_graph.initialize(left_image, right_image, 2);
        assert_eq!(4., penalty_graph.penalty(disparity_map));
    }
 }
