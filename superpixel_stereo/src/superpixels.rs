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

     #[derive(Debug)]
     /// Disparity graph is represented here
     pub struct SuperpixelRepresentation {
         /// Superpixel height
         pub height: usize,
         /// Superpixel width
         pub width: usize,
         /// Coordinates of pixels that belong to superpixel
         pub superpixels: Vec<Vec<usize>>,
         /// `L` : Left image of a stereo-pair
         pub image: Vec<Vec<u32>>,
     }

     impl SuperpixelRepresentation {
         /// Returns a superpixel representation of input image
         ///
         /// # Arguments
         ///
         /// * `image` - A 2D vector of unsigned integers that holds image
         /// * `height` - A usize value that holds height of rectangular window
         /// * `width` - A usize value that holds width of rectangular window
         pub fn initialize(image: Vec<Vec<u32>>,
                           height: usize, width:usize) -> Self {
             Self {
                 superpixels: vec![vec![0usize; image[0].len()]; image.len()],
                 image: image,
                 height: height,
                 width: width
             }
         }

         /// Returns a superpixel representation of input image.
         /// In each rectangulat window with given size (height and width) find median intensity
         /// and split all pixels in the window into two superpixels.
         /// First superpixel consists of all pixels that have intensities not bigger than median.
         /// Second superpixel consists of all other pixels from the window
         pub fn split_into_superpixels(&mut self) {
             let number_of_vertical_superpixels: usize = self.image.len() / self.height;
             let number_of_horizontal_superpixels: usize = self.image[0].len() / self.width;
             // iterate through superpixels
             for super_i in 0..number_of_vertical_superpixels {
                 for super_j in 0..number_of_horizontal_superpixels {
                     // compose vector of all intensities in the window
                     let mut intensities = Vec::new();
                     for image_i in (super_i * self.height)..(super_i * self.height + self.height) {
                         for image_j in (super_j * self.width)..(super_j * self.width + self.width) {
                             intensities.push(self.image[image_i][image_j]);
                         }
                     }
                     // find median intensity in the window
                     let median_intensity = median(&mut intensities);
                     // split pixels in the window into two superpixels
                     for image_i in (super_i * self.height)..(super_i * self.height + self.height) {
                         for image_j in (super_j * self.width)..(super_j * self.width + self.width) {
                             if self.image[image_i][image_j] <= median_intensity {
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
     }
 }
