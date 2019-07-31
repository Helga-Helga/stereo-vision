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
    use rand::Rng;
    use std::f64;
    use super::super::diffusion::diffusion::neighbor_exists;
    use super::super::diffusion::diffusion::neighbor_index;
    use super::super::diffusion::diffusion::number_of_neighbors;
    use super::super::diffusion::diffusion::approx_equal;
    use super::super::pgm_handler::pgm::pgm_writer;

    #[derive(Debug)]
    pub struct PenaltyGraph {
     pub lookup_table: Vec<Vec<f64>>,
     pub potentials: Vec<Vec<Vec<Vec<f64>>>>,
     pub dummy_potentials: Vec<Vec<Vec<Vec<f64>>>>,
     pub left_image: Vec<Vec<u32>>,
     pub right_image: Vec<Vec<u32>>,
     pub max_disparity: usize,
     pub smoothing_term: f64,
    }

    impl PenaltyGraph {
        pub fn initialize(left_image: Vec<Vec<u32>>,
                          right_image: Vec<Vec<u32>>,
                          max_disparity: usize,
                          smoothing_term: f64) -> Self {
            assert_eq!(left_image.len(), right_image.len());
            assert_eq!(left_image[0].len(), right_image[0].len());
            let mut lookup_table: Vec<Vec<f64>> = vec![vec![0f64; 256]; 256];
            for i in 0..256 {
                for j in 0..256 {
                    lookup_table[i][j] = ((i as f64) - (j as f64)).abs();
                    // lookup_table[i][j] = ((i - j) * (i - j)) as f64;
                }
            }
            Self {
                lookup_table: lookup_table,
                potentials: vec![vec![vec![vec![0f64; max_disparity]; 4]; left_image[0].len()];
                            left_image.len()],
                dummy_potentials: vec![vec![vec![vec![0f64; max_disparity]; 4]; left_image[0].len()];
                            left_image.len()],
                left_image: left_image,
                right_image: right_image,
                max_disparity: max_disparity,
                smoothing_term: smoothing_term,
            }
        }

        pub fn sum_of_potentials(&self, i: usize, j: usize, d: usize) -> f64 {
        /*
        i: number of pixel row in image
        j: number of pixel column in image
        d: disparity of pixel (i, j)
        Returns the sum of potentials between pixel (i, j) with disparity d and all its neighbors
        */
            let mut sum_of_potentials = 0.;
            for n in 0..4 {
                if neighbor_exists(i, j, n, self.left_image.len(), self.left_image[0].len()) {
                    sum_of_potentials += self.potentials[i][j][n][d];
                }
            }
            sum_of_potentials
        }

        pub fn vertex_penalty_with_potentials(&self, i: usize, j: usize, d: usize) -> f64 {
        /*
        (i, j): coordinate of pixel t in image
        d: k_t, disparity of pixel t = (i, j)
        Returns f*_t(k_t) = f_t(k_t) - sum_{t' in N(t)} phi_{tt'}(k_t), where
        t' is a neighbor of pixel t
        N(t) is a set of vertices t' that has a common edge with t
        phi_{tt'}(k_t) is a potential
        */
            self.lookup_table[self.left_image[i][j] as usize][self.right_image[i][j - d]
                as usize] - self.sum_of_potentials(i, j, d)
        }

        pub fn edge_penalty_with_potential(&self, i: usize, j: usize, n: usize, d: usize, n_d: usize) -> f64 {
        /*
        (i, j): coordinate of pixel t in image
        d: disparity of pixel t
        n: number of neighbor t' for pixel t (from 0 to 3)
        n_d: disparity of pixel t' in image
        Returns g*_{tt'}(k_t, k_t') = g_{tt'}(k_t, k_t') + phi_{tt'}(k_t) + phi_{t't}(k_t')
        */
            let (n_i, n_j, n_index) = neighbor_index(i, j, n);
            self.smoothing_term * self.lookup_table[d][n_d] + self.potentials[i][j][n][d]
                + self.potentials[n_i][n_j][n_index][n_d]
        }

        pub fn edge_exists(&self, i: usize, j: usize, n: usize, d: usize, n_d: usize) -> bool {
        /*
        (i, j): pixel coordinates
        n: neighbor number (0, 1, 2, or 3)
        d: disparity of pixel (i, j)
        n_d: disparity of n_th neighbor of pixel (i, j)
        Returns true if edge between the given vertexes exists
        */
            if neighbor_exists(i, j, n, self.left_image.len(), self.left_image[0].len()) {
                let (_n_i, n_j, _n_index) = neighbor_index(i, j, n);
                if j >= d && n_j >= n_d {
                    if n == 2 && n_d > d + 1 {
                        return false;
                    }
                    if n == 0 && d > n_d + 1 {
                        return false;
                    }
                    return true;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }

        pub fn penalty(&self, disparity_map: Vec<Vec<usize>>) -> f64 {
        /*
        disparity_map: matrix of the same size as an image,
        in cell (i, j) contains disparity of pixel (i, j)
        Returns a general penalty: sum_t f_t(k_t) + sum_{tt' in tau} g_{tt'}(k_t, k_t'), where
        f_t(k_t) is a penalty of vertex in pixel t with disparity k_t,
        tau is a set of neighbors, if tt' in tau, then they are neighbors,
        g_{tt'}(k_t, k_t') is a penalty of edge that connects
        a vertex in pixel t with disparity k_t and a neighbor pixel t' with disparity k_t'
        */
            let mut penalty: f64 = 0.;
            for i in 0..self.left_image.len() {
                for j in 0..self.left_image[0].len() {
                    if j >= disparity_map[i][j] as usize {
                        penalty += self.vertex_penalty_with_potentials(i, j, disparity_map[i][j]);
                        for n in 2..4 {
                            if neighbor_exists(i, j, n, self.left_image.len(),
                                               self.left_image[0].len()) {
                                let (n_i, n_j, _n_index) = neighbor_index(i, j, n);
                                if self.edge_exists(i, j, n, disparity_map[i][j], disparity_map[n_i][n_j]) {
                                    penalty += self.edge_penalty_with_potential(i, j, n,
                                        disparity_map[i][j], disparity_map[n_i][n_j]);
                                } else {
                                    penalty += f64::INFINITY;
                                }
                            }
                        }
                    } else {
                        penalty += f64::INFINITY;
                    }
                }
            }
            penalty
        }

        pub fn min_edge_between_neighbors(&self, i: usize, j: usize, n: usize, d: usize) -> f64 {
        /*
        (i, j): coordinate of a pixel in an image
        n: number of a neighbor (from 0 to 3)
        d: disparity of pixel t = (i, j)
        Returns min_{n_d} g_{tt'}(d, n_d), where n_d is a disparity in pixel t'
        */
            let mut min_edge: f64 = f64::INFINITY;
            for n_d in 0..self.max_disparity {
                if self.edge_exists(i, j, n, d, n_d) {
                    let current_edge = self.edge_penalty_with_potential(i, j, n, d, n_d);
                    if current_edge < min_edge {
                        min_edge = self.edge_penalty_with_potential(i, j, n, d, n_d);
                    }
                }
            }
            // assert_lt!(min_edge, f64::INFINITY);
            min_edge
        }

        pub fn update_vertex_potential(&mut self, i: usize, j: usize, n: usize, d: usize) {
        /*
        (i, j): coordinate of a pixel in an image
        n: number of a neighbor (from 0 to 3)
        d: disparity in pixel (i, j)
        Substitutes minimum edge penalties from potentials
        */
            self.dummy_potentials[i][j][n][d] -= self.min_edge_between_neighbors(i, j, n, d);
        }

        pub fn update_edge_potential(&mut self, i: usize, j: usize, n: usize, d: usize) {
        /*
        (i, j): coordinate of a pixel in an image
        n: number of a neighbor (from 0 to 3)
        d: disparity in pixel (i, j)
        Spreads the weight of the vertex on the potentials that go out of it, equally
        */
            self.dummy_potentials[i][j][n][d] += self.vertex_penalty_with_potentials(i, j, d) /
                number_of_neighbors(i, j, self.left_image.len(), self.left_image[0].len()) as f64;

            // Check if local optimum was found
            // let energy = self.energy();
            // let true_pot = self.potentials[i][j][n][d];
            // let mut rng = rand::thread_rng();
            // self.potentials[i][j][n][d] += rng.gen_range(-10., 10.);
            // assert!(self.energy() <= energy + 10E-6);
            // self.potentials[i][j][n][d] = true_pot;
        }

        pub fn diffusion_act(&mut self) {
        /*
        Updates potentials for all pixels
        Makes one diffusion iteration
        */
            for i in 0..self.left_image.len() {
                for j in 0..self.left_image[0].len() {
                    self.diffusion_act_vertexes(i, j);
                    self.diffusion_act_edges(i, j);

                    // Check if vertexes are zero after diffusion act on them
                    // for d in 0..self.max_disparity {
                    //     if j >= d {
                    //         assert!(approx_equal(self.vertex_penalty_with_potentials(i, j, d), 0., 10E-6));
                    //     }
                    // }

                    // Check if minimum edges from vertexes are equal after diffusion act
                    // for d in 0..self.max_disparity {
                    //     if j >= d {
                    //         let mut vec = Vec::new();
                    //         for n in 0..4 {
                    //             if neighbor_exists(i, j, n, self.left_image.len(), self.left_image[0].len()) {
                    //                 vec.push(self.min_edge_between_neighbors(i, j, n, d));
                    //             }
                    //         }
                    //         for i in 1..vec.len() {
                    //             assert!(approx_equal(vec[i - 1], vec[i], 10E-6));
                    //         }
                    //     }
                    // }
                }
            }
        }

        pub fn diffusion_act_vertexes(&mut self, i: usize, j: usize) {
        /*
        Updates potentials with the first way for pixel (i, j)
        */
            for n in 0..4 {
                if neighbor_exists(i, j, n, self.left_image.len(), self.left_image[0].len()) {
                    for d in 0..self.max_disparity {
                        if j >= d {
                            self.update_vertex_potential(i, j, n, d);
                        }
                    }
                }
            }
            for n in 0..4 {
                for d in 0..self.max_disparity {
                    self.potentials[i][j][n][d] = self.dummy_potentials[i][j][n][d];
                }
            }
        }

        pub fn diffusion_act_edges(&mut self, i: usize, j: usize) {
        /*
        Updates potentials with the second way for pixel (i, j)
        */
            for n in 0..4 {
                if neighbor_exists(i, j, n, self.left_image.len(), self.left_image[0].len()) {
                    for d in 0..self.max_disparity {
                        if j >= d {
                            self.update_edge_potential(i, j, n, d);
                        }
                    }
                }
            }
            for n in 0..4 {
                for d in 0..self.max_disparity {
                    self.potentials[i][j][n][d] = self.dummy_potentials[i][j][n][d];
                }
            }
        }

        pub fn diffusion(&mut self, first_iteration: usize, number_of_iterations: usize) {
        /*
        first_iteration: first iteration to start diffusion
        number_of_iterations: number of times to update potentials
        Makes diffusion iterations
        */
            let mut energy: f64 = self.energy();
            println!("Energy: {}", energy);
            let mut i = first_iteration;
            while i < first_iteration + number_of_iterations {
                println!("Iteration # {}", i);
                self.diffusion_act();
                energy = self.energy();
                println!("Energy: {}", energy);
                // let depth_map = self.build_depth_map(i);
                // self.build_left_image(depth_map, i);
                i += 1;
            }
            let depth_map = self.build_depth_map(i);
        }

        pub fn build_depth_map(&self, iteration: usize) -> Vec<Vec<usize>> {
        /*
        For now, depth map contains a disparity for each pixel that gives
        minimum vertex penalty for the pixel.
        The depth map is saved as an image to ./images/results/result_i.pgm,
        where i is an iteration number
        */
            let mut depth_map = vec![vec![0usize; self.left_image[0].len()]; self.left_image.len()];
            for i in 0..self.left_image.len() {
                for j in 0..self.left_image[0].len() {
                    depth_map[i][j] = self.min_penalty_vertex(i, j).0;
                }
            }
            let f = pgm_writer(&depth_map, format!("./images/results/result_{}.pgm", iteration), self.max_disparity);
            let _f = match f {
                Ok(file) => file,
                Err(error) => {
                    panic!("There was a problem writing a file : {:?}", error)
                },
            };
            return depth_map;
        }

        pub fn build_left_image(&self, depth_map: Vec<Vec<usize>>, iteration: usize) {
        /*
        Right image and build disparity map are used to recreate a left image.
        It is a way to test the result.
        Recreated left image should be very similar to the original one.
        It is saved as an image to ./images/results/result_left_image_i.pgm,
        where i is an iteration number
        */
            let mut left_image = vec![vec![0usize; self.left_image[0].len()]; self.left_image.len()];
            for i in 0..self.right_image.len() {
                for j in 0..self.right_image[0].len() {
                    left_image[i][j] = self.right_image[i][j - depth_map[i][j]] as usize;
                }
            }
            let f = pgm_writer(&left_image, format!("./images/results/result_left_image_{}.pgm", iteration), 255);
            let _f = match f {
                Ok(file) => file,
                Err(error) => {
                    panic!("There was a problem writing a file : {:?}", error)
                },
            };
        }

        pub fn min_penalty_vertex(&self, i: usize, j: usize) -> (usize, f64) {
        /*
        i: number of pixel row in image
        j: number of pixel column in image
        (i, j) defines pixel (its coordinate in image)
        Returns minimum penalty of given pixel (updated by potentials):
        look over each possible disparity value of pixel and choose minimum value of vertex penalty,
        as well as a disparity value used to calculate minimum penalty
        */
            let mut min_penalty_vertex: f64 = f64::INFINITY;
            let mut disparity: usize = 0;
            for d in 0..self.max_disparity {
                if j >= d {
                    let current_vertex = self.vertex_penalty_with_potentials(i, j, d);
                    if current_vertex < min_penalty_vertex {
                        disparity = d;
                        min_penalty_vertex = current_vertex;
                    }
                }
            }
            // assert_le!(min_penalty_vertex, self.vertex_penalty_with_potentials(i, j, 0));
            return (disparity, min_penalty_vertex);
        }

        pub fn min_penalty_edge(&self, i: usize, j: usize, n: usize) -> f64 {
        /*
        n: neighbor of pixel with coordinates (i, j)
        returns min_{d, d'} g*_{tt'}(d, d'), where t is pixel (i, j), t' is it neighbor,
        g*_{tt'}(d, d') = g_{tt'}(d, d') - phi_{tt'}(d) - phi_{t't}(d'),
        where phi are potentials
        So, we have fixed pixel and its neighbor;
        and search for minimum edge penalty (with potentials) between them
        based on pixel disparity and neighbor disparity
        */
            let mut min_penalty_edge: f64 = f64::INFINITY;
            for d in 0..self.max_disparity {
                for n_d in 0..self.max_disparity {
                    if self.edge_exists(i, j, n, d, n_d) {
                        let current_edge = self.edge_penalty_with_potential(i, j, n, d, n_d);
                        if current_edge < min_penalty_edge {
                            min_penalty_edge = current_edge;
                        }
                    }
                }
            }
            // assert_le!(min_penalty_edge, self.edge_penalty_with_potential(i, j, n, 0, 0));
            min_penalty_edge
        }

        pub fn energy(&self) -> f64 {
        /*
        Returns the value of energy for the problem:
        E = sum_t min_{k_t} f_t(k_t) + sum_{tt' in tau} min_{k_t, k_t'} g_{tt'}(k_t, k_t')
        It takes the most light vertices in each object
        and the most light edges between each pair of neighbor objects and
        computes a sum of its penalties
        */
            let mut energy: f64 = 0.;
            for i in 0..self.left_image.len() {
                for j in 0..self.left_image[0].len() {
                    energy += (self.min_penalty_vertex(i, j)).1;
                    for n in 2..4 {
                        if neighbor_exists(i, j, n, self.left_image.len(),
                                           self.left_image[0].len()) {
                            energy += self.min_penalty_edge(i, j, n);
                        }
                    }
                }
            }
            println!("Penalty of zero disparity map: {}", self.zero_penalty());
            println!("Penalty of zero-one disparity map: {}", self.zero_one_penalty());
            energy
        }

        pub fn zero_penalty(&self) -> f64 {
        /*
        Computes penalty of zero disparity map just to be sure,
        that at least this map doesn't change its penalty after diffusion act
        */
            let mut zero_penalty: f64 = 0.;
            for i in 0..self.left_image.len() {
                for j in 0..self.left_image[0].len() {
                    zero_penalty += self.vertex_penalty_with_potentials(i, j, 0);
                    for n in 2..4 {
                        if neighbor_exists(i, j, n, self.left_image.len(),
                                           self.left_image[0].len()) {
                            zero_penalty += self.edge_penalty_with_potential(i, j, n, 0, 0);
                        }
                    }
                }
            }
            zero_penalty
        }

        pub fn zero_one_penalty(&self) -> f64 {
        /*
        Computes penalty of disparity map where
        the first column is consists of zeros and all other columns -- of ones
        to be sure that this map doesn't change its penalty after diffusion act
        */
            let mut zero_one_penalty: f64 = 0.;
            for i in 0..self.left_image.len() {
                zero_one_penalty += self.vertex_penalty_with_potentials(i, 0, 0);
                if neighbor_exists(i, 0, 2, self.left_image.len(),
                    self.left_image[0].len()) {
                    zero_one_penalty += self.edge_penalty_with_potential(i, 0, 2, 0, 1);
                }
                if neighbor_exists(i, 0, 3, self.left_image.len(),
                    self.left_image[0].len()) {
                    zero_one_penalty += self.edge_penalty_with_potential(i, 0, 3, 0, 0);
                }
            }
            for i in 0..self.left_image.len() {
                for j in 1..self.left_image[0].len() {
                    zero_one_penalty += self.vertex_penalty_with_potentials(i, j, 1);
                    for n in 2..4 {
                        if neighbor_exists(i, j, n, self.left_image.len(),
                                           self.left_image[0].len()) {
                            zero_one_penalty += self.edge_penalty_with_potential(i, j, n, 1, 1);
                        }
                    }
                }
            }
            zero_one_penalty
        }
    }

    #[test]
    fn test_penalty_one_pixel() {
        let left_image = vec![vec![0u32; 1]; 1];
        let right_image = vec![vec![0u32; 1]; 1];
        let disparity_map = vec![vec![0usize; 1]; 1];
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, 1, 1.);
        assert_eq!(0., penalty_graph.penalty(disparity_map));
    }

    #[test]
    fn test_penalty_four_pixels_inf() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let disparity_map = vec![vec![1usize; 2]; 2];
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        assert_eq!(f64::INFINITY, penalty_graph.penalty(disparity_map));
    }

    #[test]
    fn test_penalty_four_pixels() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let disparity_map = [[0, 1].to_vec(), [0, 1].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        assert_eq!(2., penalty_graph.penalty(disparity_map));
        penalty_graph.potentials[0][0][2][0] = 1.;
        let new_disparity_map = [[0, 1].to_vec(), [0, 1].to_vec()].to_vec();
        assert_eq!(2., penalty_graph.penalty(new_disparity_map));
    }

    #[test]
    fn test_sum_of_potentials() {
        let left_image = [[1, 9].to_vec(), [5, 6].to_vec()].to_vec();
        let right_image = [[6, 3].to_vec(), [6, 6].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][2][0] = 9.4;
        penalty_graph.potentials[0][0][3][0] = -1.8;
        penalty_graph.potentials[0][1][0][0] = 6.7;
        penalty_graph.potentials[0][1][0][1] = -1.4;
        penalty_graph.potentials[1][0][1][0] = 4.1;
        assert_eq!(true, approx_equal(7.6, penalty_graph.sum_of_potentials(0, 0, 0), 10E-6));
        assert_eq!(true, approx_equal(6.7, penalty_graph.sum_of_potentials(0, 1, 0), 10E-6));
        assert_eq!(true, approx_equal(-1.4, penalty_graph.sum_of_potentials(0, 1, 1), 10E-6));
        assert_eq!(true, approx_equal(4.1, penalty_graph.sum_of_potentials(1, 0, 0), 10E-6));
        assert_eq!(true, approx_equal(0., penalty_graph.sum_of_potentials(1, 1, 0), 10E-6));
        assert_eq!(true, approx_equal(0., penalty_graph.sum_of_potentials(1, 1, 1), 10E-6));
    }

    #[test]
    fn test_vertex_penalty_with_potentials() {
        let left_image = [[166, 26, 215].to_vec(), [52, 66, 27].to_vec(), [113, 33, 214].to_vec()].to_vec();
        let right_image = [[203, 179, 158].to_vec(), [123, 160, 222].to_vec(), [90, 139, 127].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[1][1][0][0] = 0.;
        penalty_graph.potentials[1][1][1][0] = 0.;
        penalty_graph.potentials[1][1][2][0] = 1.;
        penalty_graph.potentials[1][1][3][0] = 0.1;
        penalty_graph.potentials[1][1][0][1] = 0.9;
        penalty_graph.potentials[1][1][1][1] = 0.6;
        penalty_graph.potentials[1][1][2][1] = 1.;
        penalty_graph.potentials[1][1][3][1] = 0.6;
        assert_eq!(true, approx_equal(92.9, penalty_graph.vertex_penalty_with_potentials(1, 1, 0), 10E-6));
        assert_eq!(true, approx_equal(53.9, penalty_graph.vertex_penalty_with_potentials(1, 1, 1), 10E-6));
    }

    #[test]
    fn test_edge_penalty_with_potentials() {
        let left_image = [[244, 172].to_vec()].to_vec();
        let right_image = [[168, 83].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.8;
        penalty_graph.potentials[0][1][0][0] = 0.;
        penalty_graph.potentials[0][1][0][1] = 0.1;
        assert_eq!(true, approx_equal(0.8, penalty_graph.edge_penalty_with_potential(0, 0, 2, 0, 0), 10E-6));
        assert_eq!(true, approx_equal(0.8, penalty_graph.edge_penalty_with_potential(0, 1, 0, 0, 0), 10E-6));
        assert_eq!(true, approx_equal(1.9, penalty_graph.edge_penalty_with_potential(0, 0, 2, 0, 1), 10E-6));
        assert_eq!(true, approx_equal(1.9, penalty_graph.edge_penalty_with_potential(0, 1, 0, 1, 0), 10E-6));
    }

    #[test]
    fn test_edge_exists() {
        let left_image = [[244, 172, 168, 192].to_vec(), [83, 248, 38, 204].to_vec()].to_vec();
        let right_image = [[218, 138, 65, 18].to_vec(), [225, 7, 114, 127].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 4, 1.);
        assert!(penalty_graph.edge_exists(0, 0, 2, 0, 0));
        assert!(penalty_graph.edge_exists(0, 1, 0, 0, 0));
        assert!(penalty_graph.edge_exists(0, 0, 2, 0, 1));
        assert!(penalty_graph.edge_exists(0, 1, 0, 1, 0));
        assert!(!penalty_graph.edge_exists(0, 0, 2, 1, 0));
        assert!(!penalty_graph.edge_exists(0, 0, 2, 1, 1));
        assert!(!penalty_graph.edge_exists(0, 0, 2, 2, 0));
        assert!(!penalty_graph.edge_exists(0, 1, 0, 2, 1));
        assert!(!penalty_graph.edge_exists(0, 0, 0, 0, 0));
        assert!(!penalty_graph.edge_exists(0, 3, 0, 3, 0));
        assert!(!penalty_graph.edge_exists(0, 2, 2, 0, 3));
        assert!(!penalty_graph.edge_exists(0, 0, 2, 1, 0));
    }

    #[test]
    fn min_edge_between_neighbors() {
        let left_image = [[244, 172].to_vec()].to_vec();
        let right_image = [[168, 83].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.8;
        penalty_graph.potentials[0][1][0][0] = 0.;
        penalty_graph.potentials[0][1][0][1] = 0.1;
        assert_eq!(true, approx_equal(penalty_graph.min_edge_between_neighbors(0, 0, 2, 0), 0.8, 10E-6));
        penalty_graph.potentials[0][1][0][0] = 2.;
        assert_eq!(true, approx_equal(penalty_graph.min_edge_between_neighbors(0, 0, 2, 0), 1.9, 10E-6));
    }
 }
