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
#[macro_use]
extern crate more_asserts;
extern crate rand;

mod pgm_handler;
mod diffusion;
mod penalty_graph;
mod crossing_out_graph;
mod epsilon_search;

#[cfg_attr(tarpaulin, skip)]
fn main() {
    let (right_image, r_width, r_height) =
        pgm_handler::pgm::pgm_reader("./images/corridor_r_25.pgm".to_string());
    let (left_image, l_width, l_height) =
        pgm_handler::pgm::pgm_reader("./images/corridor_l_25.pgm".to_string());
    assert_eq!(r_width, l_width);
    assert_eq!(r_height, l_height);

    let pgraph = penalty_graph::penalty_graph::PenaltyGraph::initialize(left_image, right_image, 5, 1.5);

    let vertices = vec![vec![vec![true; pgraph.max_disparity]; l_width]; l_height];
    let edges = vec![vec![vec![vec![vec![true; pgraph.max_disparity]; 4]; pgraph.max_disparity]; l_width]; l_height];
    let mut crossing_out_graph = crossing_out_graph::crossing_out_graph::CrossingOutGraph::initialize(
        pgraph, vertices, edges);

    // println!("Creating array of epsilons ...");
    // let array: Vec<f64> = epsilon_search::epsilon_search::create_array_of_epsilons(&mut crossing_out_graph, 1E-6);
    // println!("Searching for epsilon ...");
    // let epsilon: f64 = epsilon_search::epsilon_search::epsilon_search(&mut crossing_out_graph, &array);
    let epsilon: f64 = 1. / (10 * l_width * l_height) as f64;
    println!("Epsilon: {}", epsilon);

    crossing_out_graph.diffusion_while_not_consistent(epsilon, 100);

    println!("Finding disparity map ...");
    let disparity_map: Vec<Vec<usize>> = crossing_out_graph.find_best_labeling();
    println!("Disparity map is consistent: {}", diffusion::diffusion::check_disparity_map(&disparity_map));
    let f = pgm_handler::pgm::pgm_writer(&disparity_map,
                                         "images/results/best_labeling.pgm".to_string(),
                                         crossing_out_graph.penalty_graph.max_disparity);
    let _f = match f {
        Ok(file) => file,
        Err(error) => {
            panic!("There was a problem writing a file : {:?}", error)
        },
    };
    println!("Disparity map is saved to `best_labeling.pgm`");
}
