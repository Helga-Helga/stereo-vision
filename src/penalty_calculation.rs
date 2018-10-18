/*
 * MIT License
 *
 * Copyright (c) 2018 Helga-Helga
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
pub mod penalty {
    pub fn penalty_calculation(left_image: Vec<Vec<i32>>,
                                right_image: Vec<Vec<i32>>,
                                disparity_map: Vec<Vec<i32>>,
                                height: usize,
                                width: usize) -> i32 {
        let mut penalty: i32 = 0;
        for l_row in 0..height {
            for l_column in 0..width {
                if disparity_map[l_row][l_column] >= 0 {
                    if l_column + 1 < width
                    && disparity_map[l_row][l_column + 1] >= 0 {
                        penalty +=
                            (disparity_map[l_row][l_column] - disparity_map[l_row][l_column + 1])
                            .abs();
                    }
                    if l_row + 1 < height
                    && disparity_map[l_row + 1][l_column] >= 0 {
                        penalty +=
                            (disparity_map[l_row][l_column] - disparity_map[l_row + 1][l_column])
                            .abs();
                    }
                    let r_column: i32 = l_column as i32 - disparity_map[l_row][l_column] as i32;
                    if r_column > 0 {
                        penalty +=
                            (left_image[l_row][l_column] - right_image[l_row][r_column as usize])
                            .pow(2);
                    }
                }
            }
        }
        println!("Penalty: {}", penalty);
        return penalty;
    }
}
