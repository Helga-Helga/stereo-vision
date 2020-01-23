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
    use super::super::utils::utils::median;
    use super::super::utils::utils::average;

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
                image: image.to_vec()
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
            let mut left_j: usize = super_j * self.super_width + self.super_width;
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
    }
}
