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
#[doc="Superpixel"]
pub mod superpixels {
    use super::super::utils::utils::average;
    use super::super::utils::utils::median;
    use super::super::utils::utils::neighbor_exists;
    use super::super::utils::utils::neighbor_index;
    use super::super::utils::utils::neighbor_superpixel;

    #[derive(Debug)]
    /// Disparity graph is represented here
    pub struct SuperpixelRepresentation {
        /// Superpixel height
        pub super_height: usize,
        /// Superpixel width
        pub super_width: usize,
        /// Coordinates of pixels that belong to superpixel
        pub superpixels: Vec<Vec<usize>>,
        /// `L` : Left image of a stereo-pair
        pub image: Vec<Vec<u32>>,
        /// Number of superpixels in vertical axis
        pub number_of_vertical_superpixels: usize,
        /// Number of superpixels in horizontal axis
        pub number_of_horizontal_superpixels: usize,
        /// Number of pixels on edge between two superpixels
        pub edge_perimeters: Vec<Vec<Vec<Vec<usize>>>>
    }

    impl SuperpixelRepresentation {
        /// Returns a superpixel representation of input image
        ///
        /// # Arguments
        ///
        /// * `image` - A 2D vector of unsigned integers that holds image
        /// * `super_height` - A usize value that holds height of rectangular window
        /// * `super_width` - A usize value that holds width of rectangular window
        pub fn initialize(image: &Vec<Vec<u32>>,
                          super_height: usize, super_width: usize) -> Self {
            Self {
                superpixels: vec![vec![0usize; image[0].len()]; image.len()],
                super_height: super_height,
                super_width: super_width,
                number_of_vertical_superpixels: image.len() / super_height,
                number_of_horizontal_superpixels: image[0].len() / super_width,
                image: image.to_vec(),
                edge_perimeters: vec![vec![vec![vec![0usize; 9]; 2]; image[0].len()]; image.len()]
            }
        }

        /// Returns a superpixel representation of input image.
        /// In each rectangulat window with given size (height and width) find median intensity
        /// and split all pixels in the window into two superpixels.
        /// First superpixel consists of all pixels that have intensities not bigger than median.
        /// Second superpixel consists of all other pixels from the window
        pub fn split_into_superpixels(&mut self) {
            // iterate through superpixels
            for super_i in 0..self.number_of_vertical_superpixels {
                for super_j in 0..self.number_of_horizontal_superpixels {
                    // compose vector of all intensities in the window
                    let mut intensities = Vec::new();
                    for image_i in (super_i * self.super_height)..(
                                   super_i * self.super_height + self.super_height) {
                        for image_j in (super_j * self.super_width)..(
                                       super_j * self.super_width + self.super_width) {
                            intensities.push(self.image[image_i][image_j]);
                        }
                    }
                    // find median intensity in the window
                    // let median_intensity = median(&mut intensities);
                    // find average intensity in the window
                    let average_intensity = average(&intensities);
                    // split pixels in the window into two superpixels
                    for image_i in (super_i * self.super_height)..(
                                   super_i * self.super_height + self.super_height) {
                        for image_j in (super_j * self.super_width)..(
                                       super_j * self.super_width + self.super_width) {
                            if self.image[image_i][image_j] <= average_intensity {
                                self.superpixels[image_i][image_j] = 0;
                            }
                            else {
                                self.superpixels[image_i][image_j] = 1;
                            }
                        }
                    }
                }
            }
        }

        pub fn left_j_in_superpixel(&self, super_i: usize, super_j: usize,
                                    superpixel: usize) -> usize {
            let mut left_j: usize = super_j * self.super_width + self.super_width - 1;
            for image_i in (super_i * self.super_height)..(
                           super_i * self.super_height + self.super_height) {
                for image_j in (super_j * self.super_width)..(
                               super_j * self.super_width + self.super_width) {
                    if self.superpixels[image_i][image_j] == superpixel && left_j > image_j {
                        left_j = image_j;
                    }
                }
            }
            left_j
        }

        pub fn fill_superpixel_ids(&self) -> Vec<Vec<usize>> {
            let mut id: usize = 0;
            let mut superpixel_ids = vec![vec![0usize; self.image[0].len()]; self.image.len()];
            for super_i in 0..self.number_of_vertical_superpixels {
                for super_j in 0..self.number_of_horizontal_superpixels {
                    for image_i in (super_i * self.super_height)..(
                                   super_i * self.super_height + self.super_height) {
                        for image_j in (super_j * self.super_width)..(
                                       super_j * self.super_width + self.super_width) {
                            if self.superpixels[image_i][image_j] == 0 {
                                superpixel_ids[image_i][image_j] = id;
                            } else {
                                superpixel_ids[image_i][image_j] = id + 1;
                            }
                        }
                    }
                    id += 2;
                }
            }
            superpixel_ids
        }

        pub fn calculate_edge_perimeters(&mut self) {
            let superpixel_ids: Vec<Vec<usize>> = self.fill_superpixel_ids();
            for super_i in 0..self.number_of_vertical_superpixels {
                for super_j in 0..self.number_of_horizontal_superpixels {
                    let mut perimeter: usize = 0;
                    for image_i in (super_i * self.super_height)..(
                                   super_i * self.super_height + self.super_height - 1) {
                        for image_j in (super_j * self.super_width)..(
                                       super_j * self.super_width + self.super_width - 1) {
                            if superpixel_ids[image_i][image_j] !=
                                    superpixel_ids[image_i][image_j + 1]
                            || image_i + 1 < super_i * self.super_height + self.super_height
                                && superpixel_ids[image_i][image_j] !=
                                    superpixel_ids[image_i + 1][image_j] {
                                perimeter += 1;
                            }
                        }
                    }
                    for superpixel in 0..2 {
                        for n in 5..7 {
                            if !neighbor_exists(
                                super_i, super_j, n,
                                self.number_of_vertical_superpixels,
                                self.number_of_horizontal_superpixels
                            ) {
                                continue;
                            }
                            let mut perimeter_right: usize = 0;
                            let n_superpixel = neighbor_superpixel(superpixel, n);
                            let (n_i, n_j, n_index) = neighbor_index(
                                super_i, super_j, n, superpixel
                            );

                            let image_j = super_j * self.super_width + self.super_width - 1;
                            for image_i in (super_i * self.super_height)..(
                                    super_i * self.super_height + self.super_height) {
                                if self.superpixels[image_i][image_j] == superpixel
                                && self.superpixels[image_i][image_j + 1] == n_superpixel {
                                    perimeter_right += 1;
                                }
                            }
                            self.edge_perimeters[super_i][super_j][superpixel][n] =
                                perimeter_right;
                            self.edge_perimeters[n_i][n_j][n_superpixel][n_index] =
                                perimeter_right;
                        }

                        for n in 7..9 {
                            if !neighbor_exists(
                                super_i, super_j, n,
                                self.number_of_vertical_superpixels,
                                self.number_of_horizontal_superpixels
                            ) {
                                continue;
                            }
                            let mut perimeter_down: usize = 0;
                            let n_superpixel = neighbor_superpixel(superpixel, n);
                            let (n_i, n_j, n_index) = neighbor_index(
                                super_i, super_j, n, superpixel
                            );
                            let image_i = super_i * self.super_height + self.super_height - 1;
                            for image_j in (super_i * self.super_width)..(
                                    super_i * self.super_width + self.super_width) {
                                if self.superpixels[image_i][image_j] == superpixel
                                && self.superpixels[image_i + 1][image_j] == n_superpixel {
                                    perimeter_down += 1;
                                }
                            }
                            self.edge_perimeters[super_i][super_j][superpixel][n] =
                                perimeter_down;
                            self.edge_perimeters[n_i][n_j][n_superpixel][n_index] =
                                perimeter_down;
                        }
                    }
                    self.edge_perimeters[super_i][super_j][0][0] = perimeter;
                    self.edge_perimeters[super_i][super_j][1][0] = perimeter;
                }
            }
        }
    }

    #[test]
    fn test_fill_superpixel_ids() {
        let left_image = [
            [2, 2, 0, 2, 0, 0, 2, 2].to_vec(),
            [2, 2, 2, 0, 0, 2, 2, 2].to_vec(),
            [2, 2, 0, 0, 0, 2, 2, 2].to_vec(),
            [0, 0, 2, 0, 0, 0, 0, 0].to_vec()
        ].to_vec();
        let right_image = [
            [2, 2, 0, 2, 0, 0, 2, 2].to_vec(),
            [2, 2, 2, 0, 0, 2, 2, 2].to_vec(),
            [2, 2, 0, 0, 0, 2, 2, 2].to_vec(),
            [0, 0, 2, 0, 0, 0, 0, 0].to_vec()
        ].to_vec();
        let mut superpixel_representation = SuperpixelRepresentation::initialize(
            &left_image, 4, 4);
        superpixel_representation.split_into_superpixels();
        let superpixel_ids = superpixel_representation.fill_superpixel_ids();
        assert_eq!(1, superpixel_ids[0][0]);
        assert_eq!(1, superpixel_ids[0][1]);
        assert_eq!(0, superpixel_ids[0][2]);
        assert_eq!(1, superpixel_ids[0][3]);
        assert_eq!(2, superpixel_ids[0][4]);
        assert_eq!(2, superpixel_ids[0][5]);
        assert_eq!(3, superpixel_ids[0][6]);
        assert_eq!(3, superpixel_ids[0][7]);
    }

    #[test]
    fn test_calculate_edge_perimeters() {
        let left_image = [
            [2, 2, 0, 2, 0, 0, 2, 2].to_vec(),
            [2, 2, 2, 0, 0, 2, 2, 2].to_vec(),
            [2, 2, 0, 0, 0, 2, 2, 2].to_vec(),
            [0, 0, 2, 0, 0, 0, 0, 0].to_vec()
        ].to_vec();
        let right_image = [
            [2, 2, 0, 2, 0, 0, 2, 2].to_vec(),
            [2, 2, 2, 0, 0, 2, 2, 2].to_vec(),
            [2, 2, 0, 0, 0, 2, 2, 2].to_vec(),
            [0, 0, 2, 0, 0, 0, 0, 0].to_vec()
        ].to_vec();
        let mut superpixel_representation = SuperpixelRepresentation::initialize(
            &left_image, 4, 4);
        superpixel_representation.split_into_superpixels();
        superpixel_representation.calculate_edge_perimeters();
        assert_eq!(6, superpixel_representation.edge_perimeters[0][0][0][0]);
        assert_eq!(3, superpixel_representation.edge_perimeters[0][0][0][5]);
        assert_eq!(0, superpixel_representation.edge_perimeters[0][0][0][6]);
        assert_eq!(1, superpixel_representation.edge_perimeters[0][0][1][5]);
        assert_eq!(0, superpixel_representation.edge_perimeters[0][0][1][6]);
        assert_eq!(5, superpixel_representation.edge_perimeters[0][1][0][0]);
    }
}
