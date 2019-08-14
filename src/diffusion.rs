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
pub mod diffusion {
    use std::f64;

    pub fn neighbor_exists(pixel_i: usize, pixel_j: usize, neighbor: usize,
                            height: usize, width: usize) -> bool {
    /*
    pixel_i: number of current pixel row in image
    pixel_j: number of current pixel column in image
    neighbor: number of pixel neighbor (from 0 to 3)
    height: height of image
    width: width of image
    Each pixel has maximum 4 neighbors:
    - 0 for the left neighbor
    - 1 for the top neighbor
    - 2 for the right neighbor
    - 3 for the bottom neighbor
    Pixel can have less than 4 neighbors if it is in the eage of an image.
    Returns `true` if neighbor exists, `false` if not.
    */
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

    pub fn neighbor_index(i: usize, j: usize, neighbor: usize) -> (usize, usize, usize) {
    /*
    i: row of pixel in left image
    j: column of pixel in left image
    neighbor: number of pixel neighbor (from 0 to 3)
    Returns coordinates of neighbor and number of pixel for neighbor
    */
        match neighbor {
            0 => return (i, j - 1, 2),
            1 => return (i - 1, j, 3),
            2 => return (i, j + 1, 0),
            3 => return (i + 1, j, 1),
            _ => panic!("Non-existent neighbor index: {}", neighbor),
        }
    }

    pub fn number_of_neighbors(i: usize, j: usize, height: usize, width: usize) -> usize {
    /*
    (i, j): coordinate of pixel in an image
    height, width: size of image
    Returns number of neighbors for pixel (i, j).
    Maximum number of neighbors is 4, minimum is 2
    */
        let mut number_of_neighbors: usize = 0;
        for neighbor in 0..4 {
            if neighbor_exists(i, j, neighbor, height, width) {
                number_of_neighbors += 1;
            }
        }
        number_of_neighbors
    }

    pub fn approx_equal(x: f64, y: f64, epsilon: f64) -> bool {
    /*
    Returns true if x and y differ by no more than epsilon
    */
        if (x - y).abs() > epsilon {
            return false;
        }
        return true;
    }

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
}

#[cfg(test)]
mod tests {
    use super::diffusion::*;

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
}
