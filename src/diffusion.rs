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

    pub fn neighbour_exists(pixel_i: usize, pixel_j: usize, neighbour: usize,
                            height: usize, width: usize) -> bool {
    /*
    pixel_i: number of current pixel row in image
    pixel_j: number of current pixel column in image
    neighbour: number of pixel neighbour (from 0 to 3)
    height: height of image
    width: width of image
    Each pixel has maximum 4 neighbours:
    - 0 for the left neighbour
    - 1 for the top neighbour
    - 2 for the right neighbour
    - 3 for the bottom neighbour
    Pixel can have less than 4 neighbours if it is in the eage of an image.
    Returns `true` if neighbour exists, `false` if not.
    */
        match neighbour {
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
            _ => panic!(),
        }
    }

    pub fn neighbour_index(i: usize, j: usize, neighbour: usize) -> (usize, usize, usize) {
    /*
    i: row of pixel in left image
    j: column of pixel in left image
    neighbour: number of pixel neighbour (from 0 to 3)
    Returns coordinates of neighbour and number of pixel for neighbour
    */
        match neighbour {
            0 => return (i, j - 1, 2),
            1 => return (i - 1, j, 3),
            2 => return (i, j + 1, 0),
            3 => return (i + 1, j, 1),
            _ => panic!("Non-existent neighbour index: {}", neighbour),
        }
    }

    pub fn number_of_neighbours(i: usize, j: usize, height: usize, width: usize) -> usize {
    /*
    (i, j): coordinate of pixel in an image
    height, width: size of image
    Returns number of neighbours for pixel (i, j).
    Maximum number of neighbours is 4, minimum is 2
    */
        let mut number_of_neighbours: usize = 0;
        for neighbour in 0..4 {
            if neighbour_exists(i, j, neighbour, height, width) {
                number_of_neighbours += 1;
            }
        }
        number_of_neighbours
    }

    pub fn approx_equal(x: f64, y: f64) -> bool {
    /*
    Returns true if x and y are almost equal
    */
        if (x - y).abs() > f64::EPSILON {
            return false;
        }
        return true;
    }
}
