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
     const MAX_DISPARITY: usize = 5;

     pub fn diffusion(height: usize, width: usize) {

        let mut vertices = vec![vec![vec![true; MAX_DISPARITY]; width]; height];
        let mut edges =
            vec![vec![vec![vec![vec![true; MAX_DISPARITY]; 4]; MAX_DISPARITY]; width]; height];
        crossing_out(height, width, &mut vertices, &mut edges, MAX_DISPARITY);
     }

     pub fn crossing_out(height: usize,
                         width: usize,
                         vertices: &mut Vec<Vec<Vec<bool>>>,
                         edges: &mut Vec<Vec<Vec<Vec<Vec<bool>>>>>,
                         max_disparity: usize) {
        let mut changed = true;
        while changed {
            let changed_e = crossing_out_edges(height, width, &vertices, edges, max_disparity);
            let changed_v = crossing_out_vertices(height, width, vertices, &edges, max_disparity);
            changed = changed_e || changed_v;
        }
    }

     pub fn crossing_out_vertices(height: usize,
                                  width: usize,
                                  vertices: &mut Vec<Vec<Vec<bool>>>,
                                  edges: &Vec<Vec<Vec<Vec<Vec<bool>>>>>,
                                  max_disparity: usize) -> bool {
        let mut changed = false;
        for i in 0..height {
            for j in 0..width {
                for n in 0..4 {
                    if !neighbour_exists(i as i32, j as i32, n, height as i32, width as i32) {
                        continue;
                    }
                    for d in 0..max_disparity {
                        let mut edge_exists = false;
                        for d_n in 0..max_disparity {
                            if edges[i][j][d][n][d_n] {
                                edge_exists = true;
                                break;
                            }
                        }
                        if vertices[i][j][d] && !edge_exists {
                            vertices[i][j][d] = false;
                            changed = true;
                        }
                    }
                }
            }
        }
        changed
     }

     pub fn crossing_out_edges(height: usize,
                               width: usize,
                               vertices: &Vec<Vec<Vec<bool>>>,
                               edges: &mut Vec<Vec<Vec<Vec<Vec<bool>>>>>,
                               max_disparity: usize) -> bool {
         let mut changed = false;
         for i in 0..height {
             for j in 0..width {
                 for d in 0..max_disparity {
                     for n in 0..4 {
                         for d_n in 0..max_disparity {
                             if !edges[i][j][d][n][d_n] {
                                 continue;
                             }
                             if !vertices[i][j][d] {
                                 edges[i][j][d][n][d_n] = false;
                                 changed = true;
                             }
                         }
                     }
                 }
             }
         }
         changed
     }

     pub fn neighbour_exists(vertice_i: i32, vertice_j: i32, neighbour: usize,
                             height: i32, width: i32) -> bool {
         match neighbour {
             0 => if vertice_j - 1 >= 0 {
                    return true
                } else {
                    return false
                },
             1 => if vertice_i - 1 >= 0 {
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

     #[test]
     fn crossing_out_works_1() {
         let height = 2;
         let width = 2;
         let max_disparity = 2;
         let mut vertices = vec![vec![vec![true; max_disparity]; width]; height];
         let mut edges =
             vec![vec![vec![vec![vec![false; max_disparity]; 4]; max_disparity]; width]; height];
         edges[0][0][0][3][0] = true;
         edges[1][0][0][1][0] = true;
         edges[0][0][1][3][1] = true;
         edges[1][0][1][1][1] = true;
         edges[0][0][0][2][0] = true;
         edges[0][1][0][0][0] = true;
         edges[0][1][1][3][1] = true;
         edges[1][1][1][1][1] = true;
         edges[1][0][0][2][0] = true;
         edges[1][1][0][0][0] = true;
         edges[0][1][0][3][0] = true;
         edges[1][1][0][1][0] = true;
         crossing_out(height, width, &mut vertices, &mut edges, max_disparity);
         for i in 0..height {
             for j in 0..width {
                 assert_eq!(vertices[i][j][0], true);
                 assert_eq!(vertices[i][j][1], false);
             }
         }
     }
 }
