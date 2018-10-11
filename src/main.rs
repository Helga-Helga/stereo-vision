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
mod pgm_reader;
mod penalty_calculation;

fn main() {
    let (right_image, r_width, r_height) =
        pgm_reader::pgm::pgm_reader("./images/im2.pgm".to_string(), true, 0.);
    let (left_image, l_width, l_height) =
        pgm_reader::pgm::pgm_reader("./images/im6.pgm".to_string(), true, 0.);
    assert_eq!(r_width, l_width);
    assert_eq!(r_height, l_height);
    let (disparity_map , _d_width, _d_height) =
        pgm_reader::pgm::pgm_reader("./images/disp2_ascii.pgm".to_string(), false, 4.);
    assert_eq!(r_width, _d_width);
    assert_eq!(r_height, _d_height);
    let _penalty = penalty_calculation::penalty
        ::penalty_calculation(left_image, right_image, disparity_map, r_height, r_width);
}
