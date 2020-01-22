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
#[doc="Utils"]
pub mod utils {
    use std::f64;

    /// Returns `true` if requested neighbor for a given pixel exists, `false` if not.
    /// Each pixel has maximum 4 neighbors:
    /// * `0` for the left neighbor
    /// * `1` for the top neighbor
    /// * `2` for the right neighbor
    /// * `3` for the bottom neighbor
    /// Pixel can have less than `4` neighbors if it is in the edge of an image.
    ///
    /// # Arguments:
    /// * `pixel_i` - A number of current pixel row in image
    /// * `pixel_j` - A number of current pixel column in image
    /// * `neighbor` - A number of pixel neighbor (from `0` to `3`)
    /// * `height` - An image height
    /// * `width` - An image width
    pub fn neighbor_exists(pixel_i: usize, pixel_j: usize, neighbor: usize,
                            height: usize, width: usize) -> bool {
        match neighbor {
            0 => if pixel_j > 0 {
                return true
            } else {
                return false
            },
            1 => if pixel_i > 0 {
                return true
            } else {
                return false
            },
            2 => if pixel_j + 1 < width {
                return true
            } else {
                return false
            },
            3 => if pixel_i + 1 < height {
                return true
            } else {
                return false
            },
            _ => panic!("Non-existent neighbor index: {}", neighbor),
        }
    }

    /// Returns coordinates of requested neighbor and number of a given pixel as a neighbor for it
    ///
    /// # Arguments:
    /// * `i` - A row of a pixel in image
    /// * `j` - A column of a pixel in image
    /// * `neighbor` - A number of pixel neighbor
    pub fn neighbor_index(i: usize, j: usize, neighbor: usize) -> (usize, usize, usize) {
        match neighbor {
            0 => return (i, j - 1, 2),
            1 => return (i - 1, j, 3),
            2 => return (i, j + 1, 0),
            3 => return (i + 1, j, 1),
            _ => panic!("Non-existent neighbor index: {}", neighbor),
        }
    }

    /// Returns number of neighbors for a given pixel.
    /// Maximum number of neighbors is `4`, minimum is `2`
    ///
    /// # Arguments:
    /// * `i` - A number of current pixel row in image
    /// * `j` - A number of current pixel column in image
    /// * `height` - An image height
    /// * `width` - An image width
    pub fn number_of_neighbors(i: usize, j: usize, height: usize, width: usize) -> usize {
        let mut number_of_neighbors: usize = 0;
        for neighbor in 0..4 {
            if neighbor_exists(i, j, neighbor, height, width) {
                number_of_neighbors += 1;
            }
        }
        number_of_neighbors
    }

    /// Returns `true` if given numbers differ not more that by given `epsilon`
    ///
    /// # Arguments:
    /// * `x` - The first float number to compare
    /// * `y` - The second float number to compare
    /// * `epsilon` - A small float number for comparing `x` and `y`
    pub fn approx_equal(x: f64, y: f64, epsilon: f64) -> bool {
        if (x - y).abs() > epsilon {
            return false;
        }
        return true;
    }

    /// Returns `true` if a given disparity map is consistent, else - `false`.
    /// The map is consistent if there is a continuous path, connecting each pair of neighbor pixels
    ///
    /// # Arguments
    /// `disparity_map` - A matrix with disparities
    pub fn check_disparity_map(disparity_map: &Vec<Vec<usize>>) -> bool {
        for i in 0..disparity_map.len() {
            for j in 0..(disparity_map[0].len() - 1) {
                if j < disparity_map[i][j] || disparity_map[i][j + 1] > disparity_map[i][j] + 1 {
                    return false;
                }
            }
        }
        true
    }

    /// Returns sorted array of floats from the input, but without duplicates (with some precision)
    ///
    /// # Arguments
    /// * `array` - Sorted array of floats
    /// * `tolerance`: A small float value to compare values from array
    pub fn dedup_f64(array: Vec<f64>, tolerance: f64) -> Vec<f64> {
        let mut i: usize = 0;
        let mut indices_array: Vec<usize> = vec![0; array.len()];
        let mut current_index: usize = 0;
        while i < array.len() {
            indices_array[current_index] = i;
            current_index += 1;
            let mut j = i + 1;
            while j < array.len() {
                if approx_equal(array[i], array[j], tolerance) {
                    j += 1;
                } else {
                    break;
                }
            }
            i = j;
        }
        if indices_array[current_index - 1] != array.len() - 1 {
            indices_array[current_index] = array.len() - 1;
            current_index += 1;
        }
        let mut unique_values: Vec<f64> = vec![0.; current_index];
        for i in 0..current_index {
            unique_values[i] = array[indices_array[i]];
        }
        unique_values
    }

    /// Returns median value of a given vector
    ///
    /// # Arguments
    /// * `numbers` - array of u32
    pub fn median(numbers: &mut [u32]) -> u32 {
       numbers.sort();
       let mid = numbers.len() / 2;
       numbers[mid]
   }

   /// Returns average value of a given vector
   ///
   /// # Arguments
   /// * `numbers` - array of u32
   pub fn average(numbers: &[u32]) -> u32 {
       (numbers.iter().sum::<u32>() as f32 / numbers.len() as f32) as u32
   }
}

#[cfg(test)]
mod tests {
    use super::utils::*;

    #[test]
    fn test_neighbor_exists() {
        assert!(neighbor_exists(1, 1, 0, 2, 2));
        assert!(!neighbor_exists(0, 0, 0, 2, 2));
        assert!(neighbor_exists(1, 0, 1, 2, 2));
        assert!(!neighbor_exists(0, 0, 1, 2, 2));
        assert!(neighbor_exists(0, 0, 2, 2, 2));
        assert!(!neighbor_exists(0, 1, 2, 2, 2));
        assert!(neighbor_exists(0, 0, 3, 2, 2));
        assert!(!neighbor_exists(1, 0, 3, 2, 2));
    }

    #[test]
    #[should_panic]
    fn test_edge_exists_panic() {
        neighbor_exists(1, 1, 6, 2, 2);
    }

    #[test]
    fn test_neighbor_index() {
        assert_eq!(0, neighbor_index(0, 1, 0).0);
        assert_eq!(0, neighbor_index(0, 1, 0).1);
        assert_eq!(2, neighbor_index(0, 1, 0).2);

        assert_eq!(0, neighbor_index(1, 0, 1).0);
        assert_eq!(0, neighbor_index(1, 0, 1).1);
        assert_eq!(3, neighbor_index(1, 0, 1).2);

        assert_eq!(0, neighbor_index(0, 0, 2).0);
        assert_eq!(1, neighbor_index(0, 0, 2).1);
        assert_eq!(0, neighbor_index(0, 0, 2).2);

        assert_eq!(1, neighbor_index(0, 0, 3).0);
        assert_eq!(0, neighbor_index(0, 0, 3).1);
        assert_eq!(1, neighbor_index(0, 0, 3).2);
    }

    #[test]
    #[should_panic]
    fn test_neighbor_index_panic() {
        neighbor_index(0, 0, 4);
    }

    #[test]
    fn test_number_of_neighbors() {
        assert_eq!(0, number_of_neighbors(0, 0, 0, 0));
        assert_eq!(0, number_of_neighbors(0, 0, 1, 0));
        assert_eq!(0, number_of_neighbors(0, 0, 1, 1));
        assert_eq!(1, number_of_neighbors(0, 0, 1, 2));
        assert_eq!(2, number_of_neighbors(0, 0, 2, 2));
        assert_eq!(2, number_of_neighbors(0, 1, 1, 3));
        assert_eq!(3, number_of_neighbors(0, 1, 2, 3));
        assert_eq!(4, number_of_neighbors(1, 1, 3, 3));
    }

    #[test]
    fn test_approx_equal() {
        assert!(approx_equal(0., 0., 0.));
        assert!(approx_equal(1., 0., 1.));
        assert!(approx_equal(-0.1, -0.2, 0.3));
        assert!(!approx_equal(1., 0., 0.5));
        assert!(!approx_equal(-1., 1., 1.));
    }

    #[test]
    fn test_check_disparity_map() {
        let mut disparity_map = vec![vec![0usize; 1]; 1];
        assert!(check_disparity_map(&disparity_map));

        disparity_map = vec![vec![1usize; 2]; 2];
        assert!(!check_disparity_map(&disparity_map));

        disparity_map = [[0, 2, 1].to_vec()].to_vec();
        assert!(!check_disparity_map(&disparity_map));

        disparity_map = [[0, 0].to_vec(), [0, 1].to_vec()].to_vec();
        assert!(check_disparity_map(&disparity_map));
    }

    #[test]
    fn test_dedup_f64() {
        let mut array: Vec<f64> = [1., 1.5, 2., 3.].to_vec();
        array = dedup_f64(array, 1.);
        assert_eq!([1., 3.].to_vec(), array);
    }

    #[test]
    fn test_dedup_f64_one_element() {
        let mut array: Vec<f64> = [0.].to_vec();
        array = dedup_f64(array, 0.);
        assert_eq!([0.].to_vec(), array);
    }

    #[test]
    fn test_dedup_f64_no_remove() {
        let mut array: Vec<f64> = [1., 1.5, 2., 3.].to_vec();
        array = dedup_f64(array, 0.);
        assert_eq!([1., 1.5, 2., 3.].to_vec(), array);
    }
}
