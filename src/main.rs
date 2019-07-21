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
mod pgm_handler;
mod diffusion;
mod penalty_graph;
// mod crossing_out_graph;
// mod epsilon_search;

fn main() {
    let (right_image, r_width, r_height) =
        pgm_handler::pgm::pgm_reader("./images/corridor_r_25.pgm".to_string());
    let (left_image, l_width, l_height) =
        pgm_handler::pgm::pgm_reader("./images/corridor_l_25.pgm".to_string());
    assert_eq!(r_width, l_width);
    assert_eq!(r_height, l_height);
    let mut pgraph = penalty_graph::penalty_graph::PenaltyGraph::initialize(left_image, right_image, 10);
    pgraph.diffusion_while_energy_increases();
}
