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
#[doc="Diffusion graph"]
pub mod diffusion_graph {
    use rand::Rng;
    use std::f64;
    use super::super::utils::utils::neighbor_exists;
    use super::super::utils::utils::neighbor_index;
    use super::super::utils::utils::neighbor_superpixel;
    use super::super::utils::utils::number_of_neighbors;
    use super::super::utils::utils::approx_equal;
    use super::super::pgm_handler::pgm::pgm_writer;
    use super::super::superpixels::superpixels::SuperpixelRepresentation;

    #[derive(Debug)]
    /// Disparity graph is represented here
    pub struct DiffusionGraph {
        /// Lookup table for calculation of penalties easily
        pub lookup_table: Vec<Vec<f64>>,
        /// `phi` : Potentials as dual variables
        pub potentials: Vec<Vec<Vec<Vec<Vec<f64>>>>>,
        /// A copy of potentials
        pub dummy_potentials: Vec<Vec<Vec<Vec<Vec<f64>>>>>,
        /// `L` : Left image of a stereo-pair
        pub left_image: Vec<Vec<u32>>,
        /// `R` : Right image of a stereo-pair
        pub right_image: Vec<Vec<u32>>,
        /// Maximum possible disparity value to search through
        pub max_disparity: usize,
        /// Smoothing term to control the smoothness of the depth map
        pub smoothing_term: f64,
        /// Superpixel representation of left image
        pub superpixel_representation: SuperpixelRepresentation,
    }

    impl DiffusionGraph {
        /// Returns a disparity graph with given parameters
        ///
        /// # Arguments
        ///
        /// * `left_image` - A 2D vector of unsigned integers that holds left image
        /// * `right_image` - A 2D vector of unsigned integers that holds right image
        /// * `max_disparity` - A usize value that holds maximum possible disparity value
        /// * `smoothing_term` - A float value that holds smoothing term
        /// * `superpixel_representation` - Superpixel representation of left imagr
        pub fn initialize(left_image: Vec<Vec<u32>>,
                          right_image: Vec<Vec<u32>>,
                          max_disparity: usize,
                          smoothing_term: f64,
                          superpixel_representation: SuperpixelRepresentation) -> Self {
            assert_eq!(left_image.len(), right_image.len());
            assert_eq!(left_image[0].len(), right_image[0].len());
            let mut lookup_table: Vec<Vec<f64>> = vec![vec![0f64; 256]; 256];
            for i in 0..256 {
                for j in 0..256 {
                    lookup_table[i][j] = ((i as f64) - (j as f64)).abs();
                }
            }
            Self {
                lookup_table: lookup_table,
                potentials: vec![vec![vec![vec![vec![0f64; max_disparity]; 9]; 2];
                                      superpixel_representation.number_of_horizontal_superpixels];
                                 superpixel_representation.number_of_vertical_superpixels],
                dummy_potentials: vec![vec![vec![vec![vec![0f64; max_disparity]; 9]; 2];
                                            superpixel_representation.number_of_horizontal_superpixels];
                                       superpixel_representation.number_of_vertical_superpixels],
                left_image: left_image,
                right_image: right_image,
                max_disparity: max_disparity,
                smoothing_term: smoothing_term,
                superpixel_representation: superpixel_representation,
            }
        }

        /// Returns the sum of potentials between vertex in window `(super_i, super_j)`
        /// for a given superpixel with disparity d and all its neighbors
        ///
        /// <img src="https://latex.codecogs.com/png.latex?%5Csum_%7Bt%27%20%5Cin%20N%5Cleft%28t%20%5Cright%20%29%7D%20%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29">
        ///
        /// # Arguments
        ///
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `d` - A disparity value fixed in superpixel `(super_i, super_j)`
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn sum_of_potentials(&self, super_i: usize, super_j: usize, d: usize,
                                 superpixel: usize) -> f64 {
            let mut sum_of_potentials = 0.;
            for n in 0..8 {
                if neighbor_exists(super_i, super_j, n,
                                   self.superpixel_representation.number_of_vertical_superpixels,
                                   self.superpixel_representation.number_of_horizontal_superpixels) {
                    sum_of_potentials += self.potentials[super_i][super_j][superpixel][n][d];
                }
            }
            sum_of_potentials
        }

        /// Returns a vertex penalty with potentials
        ///
        /// <img src="https://latex.codecogs.com/png.latex?f_t%5Cleft%28d%20%5Cright%20%29%20%3D%20%5Cleft%7CL%5Cleft%28i%2C%20j%20%5Cright%20%29%20-%20R%5Cleft%28i%2C%20j%20-%20d%20%5Cright%20%29%20%5Cright%7C%20-%20%5Csum%5Climits_%7Bt%27%5Cin%20N%5Cleft%28t%20%5Cright%20%29%7D%20%5Cvarphi_%7Btt%27%7D%5Cleft%28d%5Cright%29">
        ///
        /// # Arguments
        ///
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `d` - A disparity value fixed in pixel `t = (super_i, super_j)`
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn vertex_penalty_with_potentials(&self, super_i: usize, super_j: usize, d: usize,
                                              superpixel: usize) -> f64 {
            let mut vertex_penalty: f64 = 0.0;
            for image_i in (super_i * self.superpixel_representation.super_height)..(
                           super_i * self.superpixel_representation.super_height + self.superpixel_representation.super_height) {
                for image_j in (super_j * self.superpixel_representation.super_width)..(
                               super_j * self.superpixel_representation.super_width + self.superpixel_representation.super_width) {
                    if self.superpixel_representation.superpixels[image_i][image_j] == superpixel {
                        vertex_penalty += self.lookup_table[self.left_image[image_i][image_j]
                            as usize][self.right_image[image_i][image_j - d] as usize];
                    }
                }
            }
            vertex_penalty - self.sum_of_potentials(super_i, super_j, d, superpixel)
        }

        /// Returns edge penalty with potentials
        ///
        /// <img src="https://latex.codecogs.com/png.latex?g_%7Btt%27%7D%5Cleft%28d%2C%20n_d%5Cright%29%20%3D%20%5Cleft%7C%20d%20-%20n_d%20%5Cright%7C%20&plus;%20%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20&plus;%20%5Cvarphi_%7Bt%27t%7D%20%5Cleft%28n_d%29">
        ///
        /// # Arguments
        ///
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (super_i, super_j)`
        /// * `n_d` - A disparity value fixed in pixel `t'`
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn edge_penalty_with_potential(&self, super_i: usize, super_j: usize, n: usize,
                                           d: usize, n_d: usize, superpixel: usize) -> f64 {
            let (n_i, n_j, n_index) = neighbor_index(super_i, super_j, n, superpixel);
            let n_superpixel = neighbor_superpixel(superpixel, n);
            self.smoothing_term * self.lookup_table[d][n_d] + self.potentials[super_i][super_j][superpixel][n][d] + self.potentials[n_i][n_j][n_superpixel][n_index][n_d]
        }

        /// Returns true if edge between the given vertexes exists
        ///
        /// # Arguments
        ///
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (super_i, super_j)`
        /// * `n_d` - A disparity value fixed in pixel `t'`
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn edge_exists(&self, super_i: usize, super_j: usize, n: usize,
                           d: usize, n_d: usize, superpixel: usize) -> bool {
            if neighbor_exists(super_i, super_j, n,
                               self.superpixel_representation.number_of_vertical_superpixels,
                               self.superpixel_representation.number_of_horizontal_superpixels) {
                let (n_i, n_j, _n_index) = neighbor_index(super_i, super_j, n, superpixel);
                let n_superpixel = neighbor_superpixel(superpixel, n);
                let left_j_in_window = self.superpixel_representation.left_j_in_superpixel(
                    super_i, super_j, superpixel);
                let left_j_in_neighbor_window = self.superpixel_representation.left_j_in_superpixel(
                    n_i, n_j, n_superpixel);
                if left_j_in_window >= d && left_j_in_neighbor_window >= n_d {
                    if (n == 5 || n == 6) && n_d > d + 1 {
                        return false;
                    }
                    if (n == 1 || n == 2) && d > n_d + 1 {
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

        /// Returns a general penalty of a given disparity map as a sum of vertice and edge penalties
        ///
        /// <img src="https://latex.codecogs.com/png.latex?P%5Cleft%28D%20%5Cright%20%29%3D%20%5Csum_t%20f_t%5Cleft%28k_t%5Cright%29%20&plus;%20%5Csum_%7Btt%27%20%5Cin%20%5Ctau%7D%20g_%7Btt%27%7D%5Cleft%28k_t%2C%20k_t%27%5Cright%29">
        ///
        /// # Arguments
        /// * `disparity_map` - A matrix of usize disparity values for the left image (with superpixels)
        pub fn penalty(&self, disparity_map: Vec<Vec<Vec<usize>>>) -> f64 {
            let mut penalty: f64 = 0.;
            for super_i in 0..self.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.superpixel_representation.number_of_horizontal_superpixels {
                    for superpixel in 0..2 {
                        let left_j_in_window = self.superpixel_representation.left_j_in_superpixel(
                            super_i, super_j, superpixel);
                        if left_j_in_window >= disparity_map[super_i][super_j][superpixel] as usize {
                            penalty += self.vertex_penalty_with_potentials(
                                super_i, super_j, disparity_map[super_i][super_j][superpixel], superpixel);
                            let possible_neighbors = [0, 5, 6, 7, 8];
                            for index in 0..possible_neighbors.len() {
                                let n = possible_neighbors[index];
                                if neighbor_exists(
                                        super_i, super_j, n,
                                        self.superpixel_representation.number_of_vertical_superpixels,
                                        self.superpixel_representation.number_of_horizontal_superpixels) {
                                    let (n_i, n_j, _n_index) = neighbor_index(
                                        super_i, super_j, n, superpixel);
                                    if self.edge_exists(super_i, super_j, n,
                                            disparity_map[super_i][super_j][superpixel],
                                            disparity_map[n_i][n_j][superpixel], superpixel) {
                                        penalty += self.edge_penalty_with_potential(super_i, super_j,
                                            n, disparity_map[super_i][super_j][superpixel],
                                            disparity_map[n_i][n_j][superpixel], superpixel);
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
            }
            penalty
        }

        /// Returns minimum edge penalty between a given vertex and neighbor
        ///
        /// <img src="https://latex.codecogs.com/png.latex?%5Cmin_%7Bn_d%7D%20g_%7Btt%27%7D%5Cleft%28d%2C%20n_d%5Cright%29">
        ///
        /// # Arguments
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (super_i, super_j)`
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn min_edge_between_neighbors(&self, super_i: usize, super_j: usize, n: usize,
                                          d: usize, superpixel: usize) -> f64 {
            let mut min_edge: f64 = f64::INFINITY;
            for n_d in 0..self.max_disparity {
                if self.edge_exists(super_i, super_j, n, d, n_d, superpixel) {
                    let current_edge = self.edge_penalty_with_potential(
                        super_i, super_j, n, d, n_d, superpixel);
                    if current_edge < min_edge {
                        min_edge = self.edge_penalty_with_potential(
                            super_i, super_j, n, d, n_d, superpixel);
                    }
                }
            }
            // assert_lt!(min_edge, f64::INFINITY);
            min_edge
        }

        /// Substitutes minimum edge penalties from potentials
        ///
        /// <img src="https://latex.codecogs.com/png.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20-%3D%20%5Cmin_%7Bn_d%7Dg_%7Btt%27%7D%5Cleft%28d%2C%20n_d%20%5Cright%20%29">
        ///
        /// # Arguments
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (super_i, super_j)`
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn update_vertex_potential(&mut self, super_i: usize, super_j: usize, n: usize,
                                       d: usize, superpixel: usize) {
            self.dummy_potentials[super_i][super_j][superpixel][n][d] -= self.min_edge_between_neighbors(
                super_i, super_j, n, d, superpixel);
        }

        /// Spreads the weight of the vertex on the  potentials that go out of it, equally
        ///
        /// <img src="https://latex.codecogs.com/png.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20&plus;%3D%20%5Cfrac%7Bf_t%5Cleft%28d%20%5Cright%20%29%7D%7B%5Cleft%7C%20N%5Cleft%28t%20%5Cright%20%29%5Cright%7C%7D">
        ///
        /// # Arguments
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (i, j)`
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn update_edge_potential(&mut self, super_i: usize, super_j: usize, n: usize, d: usize,
                                     superpixel: usize) {
            self.dummy_potentials[super_i][super_j][superpixel][n][d] += self.vertex_penalty_with_potentials(
                    super_i, super_j, d, superpixel) /
                number_of_neighbors(
                    super_i, super_j,
                    self.superpixel_representation.number_of_vertical_superpixels,
                    self.superpixel_representation.number_of_horizontal_superpixels) as f64;

            // Check if local optimum was found
            // let energy = self.energy();
            // let true_pot = self.potentials[i][j][n][d];
            // let mut rng = rand::thread_rng();
            // self.potentials[i][j][n][d] += rng.gen_range(-10., 10.);
            // assert!(self.energy() <= energy + 1E-6);
            // self.potentials[i][j][n][d] = true_pot;
        }

        /// Updates potentials for all superpixels. Makes one diffusion iteration
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20-%3D%20%5Cmin%5Climits_%7Bn_d%7D%20g_%7Btt%27%7D%5Cleft%28d%2C%20n_d%20%5Cright%20%29%2C%20%5C%2C%20%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20&plus;%3D%20%5Cmin%5Climits_%7Bn_d%7D%20%5Cfrac%7Bf_t%5Cleft%28d%20%5Cright%20%29%7D%7B%5Cleft%7CN%5Cleft%28t%20%5Cright%20%29%5Cright%7C%7D%2C%20%5C%2C%20t%20%5Cin%20T%2C%20%5C%2C%20t%27%20%5Cin%20N%5Cleft%28t%20%5Cright%20%29">
        pub fn diffusion_act(&mut self) {
            for super_i in 0..self.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.superpixel_representation.number_of_horizontal_superpixels {
                    for superpixel in 0..2 {
                        self.diffusion_act_vertexes(super_i, super_j, superpixel);
                        self.diffusion_act_edges(super_i, super_j, superpixel);
                    }

                    // Check if vertexes are zero after diffusion act on them
                    // for d in 0..self.max_disparity {
                    //     if j >= d {
                    //         assert!(approx_equal(self.vertex_penalty_with_potentials(i, j, d), 0., 1E-6));
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
                    //             assert!(approx_equal(vec[i - 1], vec[i], 1E-6));
                    //         }
                    //     }
                    // }
                }
            }
        }

        /// Updates potentials that will be used for calculation of edge penalties in superpixel
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20-%3D%20%5Cmin%5Climits_%7Bn_d%7D%20g_%7Btt%27%7D%5Cleft%28d%2C%20n_d%20%5Cright%20%29%2C%20%5C%2C%20t%27%20%5Cin%20N%5Cleft%28t%20%5Cright%20%29">
        ///
        /// # Arguments
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn diffusion_act_vertexes(&mut self, super_i: usize, super_j: usize,
                                      superpixel: usize) {
            for n in 0..9 {
                if neighbor_exists(
                        super_i, super_j, n,
                        self.superpixel_representation.number_of_vertical_superpixels,
                        self.superpixel_representation.number_of_horizontal_superpixels) {
                    for d in 0..self.max_disparity {
                        let left_j_in_window = self.superpixel_representation.left_j_in_superpixel(
                            super_i, super_j, superpixel);
                        if left_j_in_window >= d {
                            self.update_vertex_potential(super_i, super_j, n, d, superpixel);
                        }
                    }
                }
            }
            for n in 0..9 {
                for d in 0..self.max_disparity {
                    self.potentials[super_i][super_j][superpixel][n][d] = self.dummy_potentials[super_i][super_j][superpixel][n][d];
                }
            }
        }

        /// Updates potentials that will be used for calculation of vertex penalties for superpixel
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20&plus;%3D%20%5Cmin%5Climits_%7Bn_d%7D%20%5Cfrac%7Bf_t%5Cleft%28d%20%5Cright%20%29%7D%7B%5Cleft%7CN%5Cleft%28t%20%5Cright%20%29%5Cright%7C%7D%2C%20%5C%2C%20t%27%20%5Cin%20N%5Cleft%28t%20%5Cright%20%29">
        ///
        /// # Arguments
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn diffusion_act_edges(&mut self, super_i: usize, super_j: usize, superpixel: usize) {
            for n in 0..9 {
                if neighbor_exists(super_i, super_j, n,
                        self.superpixel_representation.number_of_vertical_superpixels,
                        self.superpixel_representation.number_of_horizontal_superpixels) {
                    for d in 0..self.max_disparity {
                        let left_j_in_window = self.superpixel_representation.left_j_in_superpixel(
                            super_i, super_j, superpixel);
                        if left_j_in_window >= d {
                            self.update_edge_potential(super_i, super_j, n, d, superpixel);
                        }
                    }
                }
            }
            for n in 0..9 {
                for d in 0..self.max_disparity {
                    self.potentials[super_i][super_j][superpixel][n][d] = self.dummy_potentials[
                        super_i][super_j][superpixel][n][d];
                }
            }
        }

        /// Makes diffusion iterations
        ///
        /// # Arguments
        /// * `first_iteration` - First iteration to start diffusion
        /// * `number_of_iterations` - Number of times to update potentials
        #[cfg_attr(tarpaulin, skip)]
        pub fn diffusion(&mut self, first_iteration: usize, number_of_iterations: usize) {
            let mut energy: f64 = self.energy();
            println!("Energy: {}", energy);
            let mut i = first_iteration;
            while i < first_iteration + number_of_iterations {
                println!("Iteration # {}", i);
                self.diffusion_act();
                energy = self.energy();
                println!("Energy: {}", energy);
                i += 1;
            }
            let depth_map = self.build_depth_map();
            self.build_left_image(depth_map);
        }

        /// Build and save a simple depth map from minimum vertex penalties
        ///
        /// It is saved to `./images/results/depth_map.pgm`
        pub fn build_depth_map(&self) -> Vec<Vec<usize>> {
            let mut depth_map = vec![vec![vec![0usize; 2];
                                          self.superpixel_representation.number_of_horizontal_superpixels];
                                     self.superpixel_representation.number_of_vertical_superpixels];
            let mut depth_map_image = vec![vec![0usize; self.left_image[0].len()];
                                           self.left_image.len()];
            for super_i in 0..self.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.superpixel_representation.number_of_horizontal_superpixels {
                    for superpixel in 0..2 {
                        depth_map[super_i][super_j][superpixel] = self.min_penalty_vertex(
                            super_i, super_j, superpixel).0;
                    }
                    for image_i in (super_i * self.superpixel_representation.super_height)..(
                                   super_i * self.superpixel_representation.super_height + self.superpixel_representation.super_height) {
                        for image_j in (super_j * self.superpixel_representation.super_width)..(
                                       super_j * self.superpixel_representation.super_width + self.superpixel_representation.super_width) {
                            if self.superpixel_representation.superpixels[image_i][image_j] == 0 {
                                depth_map_image[image_i][image_j] = depth_map[super_i][super_j][0];
                            } else {
                                depth_map_image[image_i][image_j] = depth_map[super_i][super_j][1];
                            }
                        }
                    }
                }
            }
            let f = pgm_writer(&depth_map_image, "./images/results/depth_map.pgm".to_string(),
                               self.max_disparity);
            let _f = match f {
                Ok(file) => file,
                Err(error) => {
                    panic!("There was a problem writing a file : {:?}", error)
                },
            };
            depth_map_image
        }

        /// Build and save left image from input right image and build disparity map
        ///
        /// It is saved to `./images/results/result_left_image.pgm`
        ///
        /// # Arguments
        /// * `depth_map` - A matrix of disparities
        pub fn build_left_image(&self, depth_map: Vec<Vec<usize>>) {
            let mut left_image = vec![vec![0usize; self.left_image[0].len()]; self.left_image.len()];
            for i in 0..self.right_image.len() {
                for j in 0..self.right_image[0].len() {
                    left_image[i][j] = self.right_image[i][j - depth_map[i][j]] as usize;
                }
            }
            let f = pgm_writer(&left_image, "./images/results/result_left_image.pgm".to_string(), 255);
            let _f = match f {
                Ok(file) => file,
                Err(error) => {
                    panic!("There was a problem writing a file : {:?}", error)
                },
            };
        }

        /// Returns minimum vertex penalty in a given pixel
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cmin%5Climits_%7Bd%7D%20f_t%5Cleft%28d%20%5Cright%20%29">
        ///
        /// # Arguments
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn min_penalty_vertex(&self, super_i: usize, super_j: usize,
                                  superpixel: usize) -> (usize, f64) {
            let mut min_penalty_vertex: f64 = f64::INFINITY;
            let mut disparity: usize = 0;
            for d in 0..self.max_disparity {
                let left_j_in_window = self.superpixel_representation.left_j_in_superpixel(
                    super_i, super_j, superpixel);
                if left_j_in_window >= d {
                    let current_vertex = self.vertex_penalty_with_potentials(super_i, super_j, d,
                                                                             superpixel);
                    if current_vertex < min_penalty_vertex {
                        disparity = d;
                        min_penalty_vertex = current_vertex;
                    }
                }
            }
            // assert_le!(min_penalty_vertex, self.vertex_penalty_with_potentials(i, j, 0));
            return (disparity, min_penalty_vertex);
        }

        /// Returns minimum edge penalty berween two neighbors
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cmin%20%5Climits_%7Bd%2C%20n_d%7D%20g_%7Btt%27%7D%28d%2C%20n_d%29">
        ///
        /// # Arguments
        /// * `super_i` - A row of a superpixel
        /// * `super_j` - A column of a superpixel
        /// * `n` - A number of pixel neighbor
        /// * `superpixel` - `0` (light) or `1` (dark). There are two superpixels in in a window
        pub fn min_penalty_edge(&self, super_i: usize, super_j: usize, n: usize,
                                superpixel: usize) -> f64 {
            let mut min_penalty_edge: f64 = f64::INFINITY;
            for d in 0..self.max_disparity {
                for n_d in 0..self.max_disparity {
                    if self.edge_exists(super_i, super_j, n, d, n_d, superpixel) {
                        let current_edge = self.edge_penalty_with_potential(
                            super_i, super_j, n, d, n_d, superpixel);
                        if current_edge < min_penalty_edge {
                            min_penalty_edge = current_edge;
                        }
                    }
                }
            }
            // assert_le!(min_penalty_edge, self.edge_penalty_with_potential(i, j, n, 0, 0));
            min_penalty_edge
        }

        /// Returns the energy value for the graph.
        /// It takes the most light vertices in each object
        /// and the most light edges between each pair of neighbor objects and
        /// computes a sum of its penalties
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?E%20%3D%20%5Csum_%7Bt%20%5Cin%20T%7D%20%5Cmin_%7Bd%7D%20f_t%28d%29%20&plus;%20%5Csum_%7Btt%27%20%5Cin%20%5Ctau%7D%20%5Cmin_%7Bd%2C%20n_d%7D%20g_%7Btt%27%7D%28d%2C%20n_d%29">
        pub fn energy(&self) -> f64 {
            let mut energy: f64 = 0.;
            for super_i in 0..self.superpixel_representation.number_of_vertical_superpixels {
                for super_j in 0..self.superpixel_representation.number_of_horizontal_superpixels {
                    for superpixel in 0..2 {
                        energy += (self.min_penalty_vertex(super_i, super_j, superpixel)).1;
                        for n in 5..9 {
                            if neighbor_exists(
                                super_i, super_j, n,
                                self.superpixel_representation.number_of_vertical_superpixels,
                                self.superpixel_representation.number_of_horizontal_superpixels) {
                                    energy += self.min_penalty_edge(super_i, super_j, n, superpixel);
                                }
                            }
                    }
                }
            }
            println!("Penalty of zero disparity map: {}", self.zero_penalty());
            println!("Penalty of zero-one disparity map: {}", self.zero_one_penalty());
            energy
        }

        /// Computes penalty of zero disparity map just to be sure,
        /// that at least this map doesn't change its penalty after diffusion act
        pub fn zero_penalty(&self) -> f64 {
            let disparity_map: Vec<Vec<Vec<usize>>> =
                vec![vec![vec![0usize; 2]; self.left_image[0].len()]; self.left_image.len()];
            self.penalty(disparity_map)
        }

        /// Computes penalty of disparity map where
        /// the first column is consists of zeros and all other columns -- of ones
        /// to be sure that this map doesn't change its penalty after diffusion act
        pub fn zero_one_penalty(&self) -> f64 {
            let mut disparity_map: Vec<Vec<Vec<usize>>> =
                vec![vec![vec![1usize; 2]; self.left_image[0].len()]; self.left_image.len()];
            for i in 0..disparity_map.len() {
                for superpixel in 0..2 {
                    disparity_map[i][0][superpixel] = 0;
                }
            }
            self.penalty(disparity_map)
        }
    }

    #[test]
    fn test_penalty_one_pixel() {
        let left_image = vec![vec![0u32; 1]; 1];
        let right_image = vec![vec![0u32; 1]; 1];
        let disparity_map = vec![vec![vec![0usize; 2]; 1]; 1];
        let mut superpixel_representation = SuperpixelRepresentation::initialize(
            &left_image, 1, 1);
        superpixel_representation.split_into_superpixels();
        let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 1, 1.,
                                                         superpixel_representation);
        assert_eq!(0., diffusion_graph.penalty(disparity_map));
    }

    #[test]
    fn test_penalty_four_pixels_inf() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let disparity_map = vec![vec![vec![1usize; 2]; 2]; 2];
        let mut superpixel_representation = SuperpixelRepresentation::initialize(
            &left_image, 1, 1);
        superpixel_representation.split_into_superpixels();
        let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.,
                                                         superpixel_representation);
        assert_eq!(f64::INFINITY, diffusion_graph.penalty(disparity_map));
    }

    #[test]
    fn test_penalty_four_pixels() {
        let left_image = [[2, 2].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[2, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let disparity_map = [[[0, 0].to_vec(), [1, 1].to_vec()].to_vec()].to_vec();
        let mut superpixel_representation = SuperpixelRepresentation::initialize(
            &left_image, 2, 2);
        superpixel_representation.split_into_superpixels();
        let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.,
                                                             superpixel_representation);
        assert_eq!(2., diffusion_graph.penalty(disparity_map));
        diffusion_graph.potentials[0][0][0][2][0] = 1.;
        diffusion_graph.potentials[0][0][1][2][0] = 1.;
        let new_disparity_map = [[[0, 0].to_vec(), [1, 1].to_vec()].to_vec(),
                                  [[0, 0].to_vec(), [1, 1].to_vec()].to_vec()].to_vec();
        assert_eq!(2., diffusion_graph.penalty(new_disparity_map));
    }

    #[test]
    fn test_penalty() {
        let left_image = [[1, 1].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec()].to_vec();
        let disparity_map = [[[0, 2].to_vec()].to_vec()].to_vec();
        let mut superpixel_representation = SuperpixelRepresentation::initialize(
            &left_image, 1, 2);
        superpixel_representation.split_into_superpixels();
        let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.,
                                                         superpixel_representation);
        assert_eq!(f64::INFINITY, diffusion_graph.penalty(disparity_map));
    }
    //
    // #[test]
    // fn test_sum_of_potentials() {
    //     let left_image = [[1, 9].to_vec(), [5, 6].to_vec()].to_vec();
    //     let right_image = [[6, 3].to_vec(), [6, 6].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][2][0] = 9.4;
    //     diffusion_graph.potentials[0][0][3][0] = -1.8;
    //     diffusion_graph.potentials[0][1][0][0] = 6.7;
    //     diffusion_graph.potentials[0][1][0][1] = -1.4;
    //     diffusion_graph.potentials[1][0][1][0] = 4.1;
    //     assert_eq!(true, approx_equal(7.6, diffusion_graph.sum_of_potentials(0, 0, 0), 1E-6));
    //     assert_eq!(true, approx_equal(6.7, diffusion_graph.sum_of_potentials(0, 1, 0), 1E-6));
    //     assert_eq!(true, approx_equal(-1.4, diffusion_graph.sum_of_potentials(0, 1, 1), 1E-6));
    //     assert_eq!(true, approx_equal(4.1, diffusion_graph.sum_of_potentials(1, 0, 0), 1E-6));
    //     assert_eq!(true, approx_equal(0., diffusion_graph.sum_of_potentials(1, 1, 0), 1E-6));
    //     assert_eq!(true, approx_equal(0., diffusion_graph.sum_of_potentials(1, 1, 1), 1E-6));
    // }
    //
    // #[test]
    // fn test_vertex_penalty_with_potentials() {
    //     let left_image = [[166, 26, 215].to_vec(), [52, 66, 27].to_vec(), [113, 33, 214].to_vec()].to_vec();
    //     let right_image = [[203, 179, 158].to_vec(), [123, 160, 222].to_vec(), [90, 139, 127].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[1][1][0][0] = 0.;
    //     diffusion_graph.potentials[1][1][1][0] = 0.;
    //     diffusion_graph.potentials[1][1][2][0] = 1.;
    //     diffusion_graph.potentials[1][1][3][0] = 0.1;
    //     diffusion_graph.potentials[1][1][0][1] = 0.9;
    //     diffusion_graph.potentials[1][1][1][1] = 0.6;
    //     diffusion_graph.potentials[1][1][2][1] = 1.;
    //     diffusion_graph.potentials[1][1][3][1] = 0.6;
    //     assert_eq!(true, approx_equal(
    //         92.9, diffusion_graph.vertex_penalty_with_potentials(1, 1, 0), 1E-6));
    //     assert_eq!(true, approx_equal(
    //         53.9, diffusion_graph.vertex_penalty_with_potentials(1, 1, 1), 1E-6));
    // }
    //
    // #[test]
    // fn test_edge_penalty_with_potentials() {
    //     let left_image = [[244, 172].to_vec()].to_vec();
    //     let right_image = [[168, 83].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][2][0] = 0.8;
    //     diffusion_graph.potentials[0][1][0][0] = 0.;
    //     diffusion_graph.potentials[0][1][0][1] = 0.1;
    //     assert_eq!(true, approx_equal(
    //         0.8, diffusion_graph.edge_penalty_with_potential(0, 0, 2, 0, 0), 1E-6));
    //     assert_eq!(true, approx_equal(
    //         0.8, diffusion_graph.edge_penalty_with_potential(0, 1, 0, 0, 0), 1E-6));
    //     assert_eq!(true, approx_equal(
    //         1.9, diffusion_graph.edge_penalty_with_potential(0, 0, 2, 0, 1), 1E-6));
    //     assert_eq!(true, approx_equal(
    //         1.9, diffusion_graph.edge_penalty_with_potential(0, 1, 0, 1, 0), 1E-6));
    // }
    //
    // #[test]
    // fn test_edge_exists() {
    //     let left_image = [[244, 172, 168, 192].to_vec(), [83, 248, 38, 204].to_vec()].to_vec();
    //     let right_image = [[218, 138, 65, 18].to_vec(), [225, 7, 114, 127].to_vec()].to_vec();
    //     let diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 4, 1.);
    //     assert!(diffusion_graph.edge_exists(0, 0, 2, 0, 0));
    //     assert!(diffusion_graph.edge_exists(0, 1, 0, 0, 0));
    //     assert!(diffusion_graph.edge_exists(0, 0, 2, 0, 1));
    //     assert!(diffusion_graph.edge_exists(0, 1, 0, 1, 0));
    //     assert!(!diffusion_graph.edge_exists(0, 0, 2, 1, 0));
    //     assert!(!diffusion_graph.edge_exists(0, 0, 2, 1, 1));
    //     assert!(!diffusion_graph.edge_exists(0, 0, 2, 2, 0));
    //     assert!(!diffusion_graph.edge_exists(0, 1, 0, 2, 1));
    //     assert!(!diffusion_graph.edge_exists(0, 0, 0, 0, 0));
    //     assert!(!diffusion_graph.edge_exists(0, 3, 0, 3, 0));
    //     assert!(!diffusion_graph.edge_exists(0, 2, 2, 0, 3));
    //     assert!(!diffusion_graph.edge_exists(0, 0, 2, 1, 0));
    // }
    //
    // #[test]
    // fn test_min_edge_between_neighbors() {
    //     let left_image = [[244, 172].to_vec()].to_vec();
    //     let right_image = [[168, 83].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][2][0] = 0.8;
    //     diffusion_graph.potentials[0][1][0][0] = 0.;
    //     diffusion_graph.potentials[0][1][0][1] = 0.1;
    //     assert_eq!(true, approx_equal(
    //         diffusion_graph.min_edge_between_neighbors(0, 0, 2, 0), 0.8, 1E-6));
    //     diffusion_graph.potentials[0][1][0][0] = 2.;
    //     assert_eq!(true, approx_equal(
    //         diffusion_graph.min_edge_between_neighbors(0, 0, 2, 0), 1.9, 1E-6));
    //     assert_eq!(f64::INFINITY, diffusion_graph.min_edge_between_neighbors(0, 0, 2, 1));
    // }
    //
    // #[test]
    // fn test_update_vertex_potential() {
    //     let left_image = [[25, 50].to_vec(), [5, 145].to_vec(), [248, 62].to_vec()].to_vec();
    //     let right_image = [[39, 15].to_vec(), [77, 145].to_vec(), [31, 71].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][3][0] = -0.3;
    //     diffusion_graph.potentials[1][0][1][0] = 2.06;
    //     diffusion_graph.potentials[1][0][2][0] = 2.06;
    //     diffusion_graph.potentials[1][0][2][1] = 2.06;
    //     diffusion_graph.potentials[1][0][3][0] = 2.06;
    //     diffusion_graph.potentials[2][0][1][0] = 0.44;
    //     diffusion_graph.potentials[1][1][0][0] = -1.42;
    //     diffusion_graph.potentials[1][1][1][0] = -1.42;
    //     diffusion_graph.potentials[1][1][0][1] = -0.62;
    //     diffusion_graph.potentials[1][1][1][1] = -0.62;
    //     diffusion_graph.potentials[1][1][3][1] = -0.62;
    //     assert_eq!(0., diffusion_graph.dummy_potentials[1][0][1][0]);
    //     diffusion_graph.update_vertex_potential(1, 0, 1, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[1][0][1][0], -1.76, 1E-6));
    //     assert_eq!(0., diffusion_graph.dummy_potentials[1][0][2][0]);
    //     diffusion_graph.update_vertex_potential(1, 0, 2, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[1][0][2][0], -0.64, 1E-6));
    //     assert_eq!(0., diffusion_graph.dummy_potentials[1][0][3][0]);
    //     diffusion_graph.update_vertex_potential(1, 0, 3, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[1][0][3][0], -2.5, 1E-6));
    // }
    //
    // #[test]
    // fn test_update_edge_potential() {
    //     let left_image = [[1, 1, 1].to_vec(), [1, 0, 1].to_vec(), [1, 1, 1].to_vec()].to_vec();
    //     let right_image = [[1, 1, 1].to_vec(), [1, 0, 1].to_vec(), [1, 1, 1].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 1, 1.);
    //     diffusion_graph.potentials[0][0][2][0] = -0.3;
    //     diffusion_graph.potentials[0][0][3][0] = 0.1;
    //     diffusion_graph.potentials[2][1][0][0] = 0.8;
    //     diffusion_graph.potentials[2][1][2][0] = -0.62;
    //     assert_eq!(0., diffusion_graph.dummy_potentials[0][0][2][0]);
    //     diffusion_graph.update_edge_potential(0, 0, 2, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[0][0][2][0], 0.1, 1E-6));
    //     diffusion_graph.update_edge_potential(1, 1, 0, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[1][1][0][0], 0., 1E-6));
    //     diffusion_graph.update_edge_potential(2, 1, 2, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[2][1][2][0], -0.06, 1E-6));
    // }
    //
    // #[test]
    // fn test_diffusion_act_vertexes() {
    //     let left_image = [[1].to_vec(), [0].to_vec()].to_vec();
    //     let right_image = [[1].to_vec(), [0].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][3][0] = -0.3;
    //     diffusion_graph.potentials[1][0][1][0] = 0.1;
    //     assert_eq!(0., diffusion_graph.dummy_potentials[0][0][3][0]);
    //     diffusion_graph.diffusion_act_vertexes(0, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[0][0][3][0], 0.2, 1E-6));
    //     assert_eq!(true, approx_equal(diffusion_graph.potentials[0][0][3][0], 0.2, 1E-6));
    //     diffusion_graph.diffusion_act_vertexes(1, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[1][0][1][0], -0.3, 1E-6));
    //     assert_eq!(true, approx_equal(diffusion_graph.potentials[1][0][1][0], -0.3, 1E-6));
    // }
    //
    // #[test]
    // fn test_diffusion_act_edges() {
    //     let left_image = [[1].to_vec(), [0].to_vec()].to_vec();
    //     let right_image = [[1].to_vec(), [0].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][3][0] = -0.3;
    //     diffusion_graph.potentials[1][0][1][0] = 0.1;
    //     assert_eq!(0., diffusion_graph.dummy_potentials[0][0][3][0]);
    //     diffusion_graph.diffusion_act_edges(0, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[0][0][3][0], 0.3, 1E-6));
    //     assert_eq!(true, approx_equal(diffusion_graph.potentials[0][0][3][0], 0.3, 1E-6));
    //     diffusion_graph.diffusion_act_edges(1, 0);
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[1][0][1][0], -0.1, 1E-6));
    //     assert_eq!(true, approx_equal(diffusion_graph.potentials[1][0][1][0], -0.1, 1E-6));
    // }
    //
    // #[test]
    // fn test_diffusion_act() {
    //     let left_image = [[1].to_vec(), [0].to_vec()].to_vec();
    //     let right_image = [[1].to_vec(), [0].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][3][0] = -0.3;
    //     diffusion_graph.potentials[1][0][1][0] = 0.1;
    //     assert_eq!(0., diffusion_graph.dummy_potentials[0][0][3][0]);
    //     diffusion_graph.diffusion_act();
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[0][0][3][0], 0., 1E-6));
    //     assert_eq!(true, approx_equal(diffusion_graph.potentials[0][0][3][0], 0., 1E-6));
    //     assert_eq!(true, approx_equal(diffusion_graph.dummy_potentials[1][0][1][0], 0., 1E-6));
    //     assert_eq!(true, approx_equal(diffusion_graph.potentials[1][0][1][0], 0., 1E-6));
    // }
    //
    // #[test]
    // fn test_min_penalty_vertex() {
    //     let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
    //     let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][2][0] = 0.6;
    //     diffusion_graph.potentials[0][0][3][0] = 358.;
    //     diffusion_graph.potentials[0][1][0][0] = -13.7;
    //     diffusion_graph.potentials[0][1][0][1] = 80.;
    //     diffusion_graph.potentials[0][1][3][1] = -1E9 as f64;
    //     assert_eq!(diffusion_graph.min_penalty_vertex(0, 0).1, -358.6);
    //     assert_eq!(0, diffusion_graph.min_penalty_vertex(0, 0).0);
    //     assert_eq!(diffusion_graph.min_penalty_vertex(0, 1).1, 14.7);
    //     assert_eq!(0, diffusion_graph.min_penalty_vertex(0, 1).0);
    // }
    //
    // #[test]
    // fn test_min_penalty_edge() {
    //     let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
    //     let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][2][0] = 0.6;
    //     diffusion_graph.potentials[0][0][3][0] = 358.;
    //     diffusion_graph.potentials[0][1][0][0] = -13.7;
    //     diffusion_graph.potentials[0][1][0][1] = 80.;
    //     diffusion_graph.potentials[0][1][3][1] = -1E9 as f64;
    //     assert_eq!(-13.1, diffusion_graph.min_penalty_edge(0, 0, 2));
    //     assert_eq!(-13.1, diffusion_graph.min_penalty_edge(0, 1, 0));
    // }
    //
    // #[test]
    // fn test_energy() {
    //     let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
    //     let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
    //     let mut diffusion_graph = DiffusionGraph::initialize(left_image, right_image, 2, 1.);
    //     diffusion_graph.potentials[0][0][2][0] = 0.6;
    //     diffusion_graph.potentials[0][0][3][0] = 358.;
    //     diffusion_graph.potentials[0][1][0][0] = -13.7;
    //     diffusion_graph.potentials[0][1][0][1] = 80.;
    //     diffusion_graph.potentials[0][1][3][1] = -1E9 as f64;
    //     assert_eq!(-999999999.0, diffusion_graph.energy());
    //     assert_eq!(0., diffusion_graph.potentials[0][0][1][0]);
    //     assert_eq!(0., diffusion_graph.potentials[0][0][1][1]);
    //     assert_eq!(0., diffusion_graph.potentials[0][1][1][1]);
    // }
 }
