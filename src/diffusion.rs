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
     pub fn neighbour_exists(vertice_i: usize, vertice_j: usize, neighbour: usize,
                             height: usize, width: usize) -> bool {
         match neighbour {
             0 => if vertice_j > 0 {
                    return true
                } else {
                    return false
                },
             1 => if vertice_i > 0 {
                    return true
                } else {
                    return false
                },
             2 => if vertice_j + 1 < width {
                     return true
                } else {
                    return false
                },
             3 => if vertice_i + 1 < height {
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
 }
