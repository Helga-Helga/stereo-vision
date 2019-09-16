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
#[doc="Disparity graph"]
pub mod penalty_graph {
    use rand::Rng;
    use std::f64;
    use super::super::diffusion::diffusion::neighbor_exists;
    use super::super::diffusion::diffusion::neighbor_index;
    use super::super::diffusion::diffusion::number_of_neighbors;
    use super::super::diffusion::diffusion::approx_equal;
    use super::super::pgm_handler::pgm::pgm_writer;

    #[derive(Debug)]
    /// Disparity graph is represented here
    pub struct PenaltyGraph {
        /// Lookup table for calculation of penalties easily
        pub lookup_table: Vec<Vec<f64>>,
        /// `phi` : Potentials as dual variables
        pub potentials: Vec<Vec<Vec<Vec<f64>>>>,
        /// A copy of potentials
        pub dummy_potentials: Vec<Vec<Vec<Vec<f64>>>>,
        /// `L` : Left image of a stereo-pair
        pub left_image: Vec<Vec<u32>>,
        /// `R` : Right image of a stereo-pair
        pub right_image: Vec<Vec<u32>>,
        /// Maximum possible disparity value to search through
        pub max_disparity: usize,
        /// Smoothing term to control the smoothness of the depth map
        pub smoothing_term: f64,
    }

    impl PenaltyGraph {
        /// Returns a disparity graph with given parameters
        ///
        /// # Arguments
        ///
        /// * `left_image` - A 2D vector of unsigned integers that holds left image
        /// * `right_image` - A 2D vector of unsigned integers that holds right image
        /// * `max_disparity` - A usize value that holds maximum possible disparity value
        /// * `smoothing_term` - A float value that holds smoothing term
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

        /// Returns the sum of potentials between vertex in pixel `(i, j)` with disparity d and all its neighbors
        ///
        /// <img src="https://latex.codecogs.com/png.latex?%5Csum_%7Bt%27%20%5Cin%20N%5Cleft%28t%20%5Cright%20%29%7D%20%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29">
        ///
        /// # Arguments
        ///
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        /// * `d` - A disparity value fixed in pixel `(i, j)`
        pub fn sum_of_potentials(&self, i: usize, j: usize, d: usize) -> f64 {
            let mut sum_of_potentials = 0.;
            for n in 0..4 {
                if neighbor_exists(i, j, n, self.left_image.len(), self.left_image[0].len()) {
                    sum_of_potentials += self.potentials[i][j][n][d];
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
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        /// * `d` - A disparity value fixed in pixel `t = (i, j)`
        pub fn vertex_penalty_with_potentials(&self, i: usize, j: usize, d: usize) -> f64 {
            self.lookup_table[self.left_image[i][j] as usize][self.right_image[i][j - d]
                as usize] - self.sum_of_potentials(i, j, d)
        }

        /// Returns edge penalty with potentials
        ///
        /// <img src="https://latex.codecogs.com/png.latex?g_%7Btt%27%7D%5Cleft%28d%2C%20n_d%5Cright%29%20%3D%20%5Cleft%7C%20d%20-%20n_d%20%5Cright%7C%20&plus;%20%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20&plus;%20%5Cvarphi_%7Bt%27t%7D%20%5Cleft%28n_d%29">
        ///
        /// # Arguments
        ///
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (i, j)`
        /// * `n_d` - A disparity value fixed in pixel `t'`
        pub fn edge_penalty_with_potential(&self, i: usize, j: usize, n: usize, d: usize, n_d: usize) -> f64 {
            let (n_i, n_j, n_index) = neighbor_index(i, j, n);
            self.smoothing_term * self.lookup_table[d][n_d] + self.potentials[i][j][n][d]
                + self.potentials[n_i][n_j][n_index][n_d]
        }

        /// Returns true if edge between the given vertexes exists
        ///
        /// # Arguments
        ///
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (i, j)`
        /// * `n_d` - A disparity value fixed in pixel `t'`
        pub fn edge_exists(&self, i: usize, j: usize, n: usize, d: usize, n_d: usize) -> bool {
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

        /// Returns a general penalty of a given disparity map as a sum of vertice and edge penalties
        ///
        /// <img src="https://latex.codecogs.com/png.latex?P%5Cleft%28D%20%5Cright%20%29%3D%20%5Csum_t%20f_t%5Cleft%28k_t%5Cright%29%20&plus;%20%5Csum_%7Btt%27%20%5Cin%20%5Ctau%7D%20g_%7Btt%27%7D%5Cleft%28k_t%2C%20k_t%27%5Cright%29">
        ///
        /// # Arguments
        /// * `disparity_map` - A matrix of usize disparity values for the left image
        pub fn penalty(&self, disparity_map: Vec<Vec<usize>>) -> f64 {
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

        /// Returns minimum edge penalty between a given vertex and neighbor
        ///
        /// <img src="https://latex.codecogs.com/png.latex?%5Cmin_%7Bn_d%7D%20g_%7Btt%27%7D%5Cleft%28d%2C%20n_d%5Cright%29">
        ///
        /// # Arguments
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (i, j)`
        pub fn min_edge_between_neighbors(&self, i: usize, j: usize, n: usize, d: usize) -> f64 {
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

        /// Substitutes minimum edge penalties from potentials
        ///
        /// <img src="https://latex.codecogs.com/png.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20-%3D%20%5Cmin_%7Bn_d%7Dg_%7Btt%27%7D%5Cleft%28d%2C%20n_d%20%5Cright%20%29">
        ///
        /// # Arguments
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (i, j)`
        pub fn update_vertex_potential(&mut self, i: usize, j: usize, n: usize, d: usize) {
            self.dummy_potentials[i][j][n][d] -= self.min_edge_between_neighbors(i, j, n, d);
        }

        /// Spreads the weight of the vertex on the  potentials that go out of it, equally
        ///
        /// <img src="https://latex.codecogs.com/png.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20&plus;%3D%20%5Cfrac%7Bf_t%5Cleft%28d%20%5Cright%20%29%7D%7B%5Cleft%7C%20N%5Cleft%28t%20%5Cright%20%29%5Cright%7C%7D">
        ///
        /// # Arguments
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        /// * `n` - A number of pixel neighbor
        /// * `d` - A disparity value fixed in pixel `t = (i, j)`
        pub fn update_edge_potential(&mut self, i: usize, j: usize, n: usize, d: usize) {
            self.dummy_potentials[i][j][n][d] += self.vertex_penalty_with_potentials(i, j, d) /
                number_of_neighbors(i, j, self.left_image.len(), self.left_image[0].len()) as f64;

            // Check if local optimum was found
            // let energy = self.energy();
            // let true_pot = self.potentials[i][j][n][d];
            // let mut rng = rand::thread_rng();
            // self.potentials[i][j][n][d] += rng.gen_range(-10., 10.);
            // assert!(self.energy() <= energy + 1E-6);
            // self.potentials[i][j][n][d] = true_pot;
        }

        /// Updates potentials for all pixels. Makes one diffusion iteration
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20-%3D%20%5Cmin%5Climits_%7Bn_d%7D%20g_%7Btt%27%7D%5Cleft%28d%2C%20n_d%20%5Cright%20%29%2C%20%5C%2C%20%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20&plus;%3D%20%5Cmin%5Climits_%7Bn_d%7D%20%5Cfrac%7Bf_t%5Cleft%28d%20%5Cright%20%29%7D%7B%5Cleft%7CN%5Cleft%28t%20%5Cright%20%29%5Cright%7C%7D%2C%20%5C%2C%20t%20%5Cin%20T%2C%20%5C%2C%20t%27%20%5Cin%20N%5Cleft%28t%20%5Cright%20%29">
        pub fn diffusion_act(&mut self) {
            for i in 0..self.left_image.len() {
                for j in 0..self.left_image[0].len() {
                    self.diffusion_act_vertexes(i, j);
                    self.diffusion_act_edges(i, j);

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

        /// Updates potentials that will be used for calculation of edge penalties the given pixel
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20-%3D%20%5Cmin%5Climits_%7Bn_d%7D%20g_%7Btt%27%7D%5Cleft%28d%2C%20n_d%20%5Cright%20%29%2C%20%5C%2C%20t%27%20%5Cin%20N%5Cleft%28t%20%5Cright%20%29">
        ///
        /// # Arguments
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        pub fn diffusion_act_vertexes(&mut self, i: usize, j: usize) {
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

        /// Updates potentials that will be used for calculation of vertex penalties for the given pixel
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cvarphi_%7Btt%27%7D%5Cleft%28d%20%5Cright%20%29%20&plus;%3D%20%5Cmin%5Climits_%7Bn_d%7D%20%5Cfrac%7Bf_t%5Cleft%28d%20%5Cright%20%29%7D%7B%5Cleft%7CN%5Cleft%28t%20%5Cright%20%29%5Cright%7C%7D%2C%20%5C%2C%20t%27%20%5Cin%20N%5Cleft%28t%20%5Cright%20%29">
        ///
        /// # Arguments
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        pub fn diffusion_act_edges(&mut self, i: usize, j: usize) {
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
                // self.build_depth_map(i);
                // self.build_left_image(depth_map, i);
                i += 1;
            }
            self.build_depth_map(i);
        }

        /// Build and save a simple depth map from minimum vertex penalties
        ///
        /// It is saved to `./images/results/result_{iteration}.pgm`
        ///
        /// # Arguments
        /// * `iteration` - A number of a current iteration. Needed for image name
        pub fn build_depth_map(&self, iteration: usize) {
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
        }

        /// Build and save left image from input right image and build disparity map
        ///
        /// It is saved to `./images/results/result_left_image_{iteration}.pgm`
        ///
        /// # Arguments
        /// * `depth_map` - A matrix of disparities
        /// * `iteration` - A number of a current iteration. Needed for image name
        pub fn build_left_image(&self, depth_map: Vec<Vec<usize>>, iteration: usize) {
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

        /// Returns minimum vertex penalty in a given pixel
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cmin%5Climits_%7Bd%7D%20f_t%5Cleft%28d%20%5Cright%20%29">
        ///
        /// # Arguments
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        pub fn min_penalty_vertex(&self, i: usize, j: usize) -> (usize, f64) {
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

        /// Returns minimum edge penalty berween two neighbors
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?%5Cmin%20%5Climits_%7Bd%2C%20n_d%7D%20g_%7Btt%27%7D%28d%2C%20n_d%29">
        ///
        /// # Arguments
        /// * `i` - A row of a pixel in image
        /// * `j` - A column of a pixel in image
        /// * `n` - A number of pixel neighbor
        pub fn min_penalty_edge(&self, i: usize, j: usize, n: usize) -> f64 {
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

        /// Returns the energy value for the graph.
        /// It takes the most light vertices in each object
        /// and the most light edges between each pair of neighbor objects and
        /// computes a sum of its penalties
        ///
        /// <img src="https://latex.codecogs.com/gif.latex?E%20%3D%20%5Csum_%7Bt%20%5Cin%20T%7D%20%5Cmin_%7Bd%7D%20f_t%28d%29%20&plus;%20%5Csum_%7Btt%27%20%5Cin%20%5Ctau%7D%20%5Cmin_%7Bd%2C%20n_d%7D%20g_%7Btt%27%7D%28d%2C%20n_d%29">
        pub fn energy(&self) -> f64 {
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

        /// Computes penalty of zero disparity map just to be sure,
        /// that at least this map doesn't change its penalty after diffusion act
        pub fn zero_penalty(&self) -> f64 {
            let disparity_map: Vec<Vec<usize>> =
                vec![vec![0usize; self.left_image[0].len()]; self.left_image.len()];
            self.penalty(disparity_map)
        }

        /// Computes penalty of disparity map where
        /// the first column is consists of zeros and all other columns -- of ones
        /// to be sure that this map doesn't change its penalty after diffusion act
        pub fn zero_one_penalty(&self) -> f64 {
            let mut disparity_map: Vec<Vec<usize>> =
                vec![vec![1usize; self.left_image[0].len()]; self.left_image.len()];
            for i in 0..disparity_map.len() {
                disparity_map[i][0] = 0;
            }
            self.penalty(disparity_map)
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
    fn test_penalty() {
        let left_image = [[1, 1].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec()].to_vec();
        let disparity_map = [[0, 2].to_vec()].to_vec();
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        assert_eq!(f64::INFINITY, penalty_graph.penalty(disparity_map));
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
        assert_eq!(true, approx_equal(7.6, penalty_graph.sum_of_potentials(0, 0, 0), 1E-6));
        assert_eq!(true, approx_equal(6.7, penalty_graph.sum_of_potentials(0, 1, 0), 1E-6));
        assert_eq!(true, approx_equal(-1.4, penalty_graph.sum_of_potentials(0, 1, 1), 1E-6));
        assert_eq!(true, approx_equal(4.1, penalty_graph.sum_of_potentials(1, 0, 0), 1E-6));
        assert_eq!(true, approx_equal(0., penalty_graph.sum_of_potentials(1, 1, 0), 1E-6));
        assert_eq!(true, approx_equal(0., penalty_graph.sum_of_potentials(1, 1, 1), 1E-6));
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
        assert_eq!(true, approx_equal(92.9, penalty_graph.vertex_penalty_with_potentials(1, 1, 0), 1E-6));
        assert_eq!(true, approx_equal(53.9, penalty_graph.vertex_penalty_with_potentials(1, 1, 1), 1E-6));
    }

    #[test]
    fn test_edge_penalty_with_potentials() {
        let left_image = [[244, 172].to_vec()].to_vec();
        let right_image = [[168, 83].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.8;
        penalty_graph.potentials[0][1][0][0] = 0.;
        penalty_graph.potentials[0][1][0][1] = 0.1;
        assert_eq!(true, approx_equal(0.8, penalty_graph.edge_penalty_with_potential(0, 0, 2, 0, 0), 1E-6));
        assert_eq!(true, approx_equal(0.8, penalty_graph.edge_penalty_with_potential(0, 1, 0, 0, 0), 1E-6));
        assert_eq!(true, approx_equal(1.9, penalty_graph.edge_penalty_with_potential(0, 0, 2, 0, 1), 1E-6));
        assert_eq!(true, approx_equal(1.9, penalty_graph.edge_penalty_with_potential(0, 1, 0, 1, 0), 1E-6));
    }

    #[test]
    fn test_edge_exists() {
        let left_image = [[244, 172, 168, 192].to_vec(), [83, 248, 38, 204].to_vec()].to_vec();
        let right_image = [[218, 138, 65, 18].to_vec(), [225, 7, 114, 127].to_vec()].to_vec();
        let penalty_graph = PenaltyGraph::initialize(left_image, right_image, 4, 1.);
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
    fn test_min_edge_between_neighbors() {
        let left_image = [[244, 172].to_vec()].to_vec();
        let right_image = [[168, 83].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.8;
        penalty_graph.potentials[0][1][0][0] = 0.;
        penalty_graph.potentials[0][1][0][1] = 0.1;
        assert_eq!(true, approx_equal(penalty_graph.min_edge_between_neighbors(0, 0, 2, 0), 0.8, 1E-6));
        penalty_graph.potentials[0][1][0][0] = 2.;
        assert_eq!(true, approx_equal(penalty_graph.min_edge_between_neighbors(0, 0, 2, 0), 1.9, 1E-6));
        assert_eq!(f64::INFINITY, penalty_graph.min_edge_between_neighbors(0, 0, 2, 1));
    }

    #[test]
    fn test_update_vertex_potential() {
        let left_image = [[25, 50].to_vec(), [5, 145].to_vec(), [248, 62].to_vec()].to_vec();
        let right_image = [[39, 15].to_vec(), [77, 145].to_vec(), [31, 71].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][3][0] = -0.3;
        penalty_graph.potentials[1][0][1][0] = 2.06;
        penalty_graph.potentials[1][0][2][0] = 2.06;
        penalty_graph.potentials[1][0][2][1] = 2.06;
        penalty_graph.potentials[1][0][3][0] = 2.06;
        penalty_graph.potentials[2][0][1][0] = 0.44;
        penalty_graph.potentials[1][1][0][0] = -1.42;
        penalty_graph.potentials[1][1][1][0] = -1.42;
        penalty_graph.potentials[1][1][0][1] = -0.62;
        penalty_graph.potentials[1][1][1][1] = -0.62;
        penalty_graph.potentials[1][1][3][1] = -0.62;
        assert_eq!(0., penalty_graph.dummy_potentials[1][0][1][0]);
        penalty_graph.update_vertex_potential(1, 0, 1, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[1][0][1][0], -1.76, 1E-6));
        assert_eq!(0., penalty_graph.dummy_potentials[1][0][2][0]);
        penalty_graph.update_vertex_potential(1, 0, 2, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[1][0][2][0], -0.64, 1E-6));
        assert_eq!(0., penalty_graph.dummy_potentials[1][0][3][0]);
        penalty_graph.update_vertex_potential(1, 0, 3, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[1][0][3][0], -2.5, 1E-6));
    }

    #[test]
    fn test_update_edge_potential() {
        let left_image = [[1, 1, 1].to_vec(), [1, 0, 1].to_vec(), [1, 1, 1].to_vec()].to_vec();
        let right_image = [[1, 1, 1].to_vec(), [1, 0, 1].to_vec(), [1, 1, 1].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 1, 1.);
        penalty_graph.potentials[0][0][2][0] = -0.3;
        penalty_graph.potentials[0][0][3][0] = 0.1;
        penalty_graph.potentials[2][1][0][0] = 0.8;
        penalty_graph.potentials[2][1][2][0] = -0.62;
        assert_eq!(0., penalty_graph.dummy_potentials[0][0][2][0]);
        penalty_graph.update_edge_potential(0, 0, 2, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[0][0][2][0], 0.1, 1E-6));
        penalty_graph.update_edge_potential(1, 1, 0, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[1][1][0][0], 0., 1E-6));
        penalty_graph.update_edge_potential(2, 1, 2, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[2][1][2][0], -0.06, 1E-6));
    }

    #[test]
    fn test_diffusion_act_vertexes() {
        let left_image = [[1].to_vec(), [0].to_vec()].to_vec();
        let right_image = [[1].to_vec(), [0].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][3][0] = -0.3;
        penalty_graph.potentials[1][0][1][0] = 0.1;
        assert_eq!(0., penalty_graph.dummy_potentials[0][0][3][0]);
        penalty_graph.diffusion_act_vertexes(0, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[0][0][3][0], 0.2, 1E-6));
        assert_eq!(true, approx_equal(penalty_graph.potentials[0][0][3][0], 0.2, 1E-6));
        penalty_graph.diffusion_act_vertexes(1, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[1][0][1][0], -0.3, 1E-6));
        assert_eq!(true, approx_equal(penalty_graph.potentials[1][0][1][0], -0.3, 1E-6));
    }

    #[test]
    fn test_diffusion_act_edges() {
        let left_image = [[1].to_vec(), [0].to_vec()].to_vec();
        let right_image = [[1].to_vec(), [0].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][3][0] = -0.3;
        penalty_graph.potentials[1][0][1][0] = 0.1;
        assert_eq!(0., penalty_graph.dummy_potentials[0][0][3][0]);
        penalty_graph.diffusion_act_edges(0, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[0][0][3][0], 0.3, 1E-6));
        assert_eq!(true, approx_equal(penalty_graph.potentials[0][0][3][0], 0.3, 1E-6));
        penalty_graph.diffusion_act_edges(1, 0);
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[1][0][1][0], -0.1, 1E-6));
        assert_eq!(true, approx_equal(penalty_graph.potentials[1][0][1][0], -0.1, 1E-6));
    }

    #[test]
    fn test_diffusion_act() {
        let left_image = [[1].to_vec(), [0].to_vec()].to_vec();
        let right_image = [[1].to_vec(), [0].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][3][0] = -0.3;
        penalty_graph.potentials[1][0][1][0] = 0.1;
        assert_eq!(0., penalty_graph.dummy_potentials[0][0][3][0]);
        penalty_graph.diffusion_act();
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[0][0][3][0], 0., 1E-6));
        assert_eq!(true, approx_equal(penalty_graph.potentials[0][0][3][0], 0., 1E-6));
        assert_eq!(true, approx_equal(penalty_graph.dummy_potentials[1][0][1][0], 0., 1E-6));
        assert_eq!(true, approx_equal(penalty_graph.potentials[1][0][1][0], 0., 1E-6));
    }

    #[test]
    fn test_min_penalty_vertex() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.6;
        penalty_graph.potentials[0][0][3][0] = 358.;
        penalty_graph.potentials[0][1][0][0] = -13.7;
        penalty_graph.potentials[0][1][0][1] = 80.;
        penalty_graph.potentials[0][1][3][1] = -1E9 as f64;
        assert_eq!(penalty_graph.min_penalty_vertex(0, 0).1, -358.6);
        assert_eq!(0, penalty_graph.min_penalty_vertex(0, 0).0);
        assert_eq!(penalty_graph.min_penalty_vertex(0, 1).1, 14.7);
        assert_eq!(0, penalty_graph.min_penalty_vertex(0, 1).0);
    }

    #[test]
    fn test_min_penalty_edge() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.6;
        penalty_graph.potentials[0][0][3][0] = 358.;
        penalty_graph.potentials[0][1][0][0] = -13.7;
        penalty_graph.potentials[0][1][0][1] = 80.;
        penalty_graph.potentials[0][1][3][1] = -1E9 as f64;
        assert_eq!(-13.1, penalty_graph.min_penalty_edge(0, 0, 2));
        assert_eq!(-13.1, penalty_graph.min_penalty_edge(0, 1, 0));
    }

    #[test]
    fn test_energy() {
        let left_image = [[1, 1].to_vec(), [0, 0].to_vec()].to_vec();
        let right_image = [[1, 0].to_vec(), [0, 0].to_vec()].to_vec();
        let mut penalty_graph = PenaltyGraph::initialize(left_image, right_image, 2, 1.);
        penalty_graph.potentials[0][0][2][0] = 0.6;
        penalty_graph.potentials[0][0][3][0] = 358.;
        penalty_graph.potentials[0][1][0][0] = -13.7;
        penalty_graph.potentials[0][1][0][1] = 80.;
        penalty_graph.potentials[0][1][3][1] = -1E9 as f64;
        assert_eq!(-999999999.0, penalty_graph.energy());
        assert_eq!(0., penalty_graph.potentials[0][0][1][0]);
        assert_eq!(0., penalty_graph.potentials[0][0][1][1]);
        assert_eq!(0., penalty_graph.potentials[0][1][1][1]);
    }
 }
