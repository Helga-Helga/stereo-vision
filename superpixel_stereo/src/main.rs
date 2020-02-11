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
extern crate tempdir;

pub mod pgm_handler;
pub mod utils;
pub mod diffusion_graph;
pub mod crossing_out_graph;
// pub mod epsilon_search;
pub mod superpixels;

#[cfg_attr(tarpaulin, skip)]
fn main() {
    let (right_image, r_width, r_height) =
        pgm_handler::pgm::pgm_reader("./images/right_teddy_original.pgm".to_string());
    let (left_image, l_width, l_height) =
        pgm_handler::pgm::pgm_reader("./images/left_teddy_original.pgm".to_string());
    assert_eq!(r_width, l_width);
    assert_eq!(r_height, l_height);

    let mut superpixel_representation = superpixels::superpixels::SuperpixelRepresentation
        ::initialize(&left_image, 8, 5);
    superpixel_representation.split_into_superpixels();
    let f = pgm_handler::pgm::pgm_writer(
        &superpixel_representation.superpixels,
        "images/results/superpixels.pgm".to_string(),
        1
    );
    let _f = match f {
        Ok(file) => file,
        Err(error) => {
            panic!("There was a problem writing a file : {:?}", error)
        },
    };
    println!("Superpixel representation of left image is saved to `superpixels.pgm`");

    let max_disparity = 30;
    println!("max disparity: {}", max_disparity);

    let mut diffusion_graph = diffusion_graph::diffusion_graph::DiffusionGraph::initialize(
        left_image, right_image, max_disparity, 1.2, superpixel_representation);

    let vertices = vec![vec![vec![vec![true; diffusion_graph.max_disparity]; 2]; l_width];
                        l_height];
    let edges = vec![vec![vec![vec![vec![vec![true; diffusion_graph.max_disparity]; 9];
                                         diffusion_graph.max_disparity]; 2]; l_width]; l_height];
    let mut crossing_out_graph = crossing_out_graph::crossing_out_graph::CrossingOutGraph
        ::initialize(diffusion_graph, vertices, edges);

    let epsilon: f64 = 1. / (10 * l_width * l_height) as f64;
    println!("Epsilon: {}", epsilon);

    crossing_out_graph.diffusion_while_not_consistent(epsilon, 100);

    println!("Finding disparity map ...");
    let disparity_map: Vec<Vec<usize>> = crossing_out_graph.simple_best_labeling();
    println!(
        "Disparity map is consistent: {}",
        utils::utils::check_disparity_map_consistency(&disparity_map));
    let f = pgm_handler::pgm::pgm_writer(
        &disparity_map,
        "images/results/best_labeling.pgm".to_string(),
        crossing_out_graph.diffusion_graph.max_disparity
    );
    let _f = match f {
        Ok(file) => file,
        Err(error) => {
            panic!("There was a problem writing a file : {:?}", error)
        },
    };
    println!("Disparity map is saved to `best_labeling.pgm`");
}
