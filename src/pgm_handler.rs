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
pub mod pgm {
    use std::fs::File;
    use std::io::{BufRead, BufReader, Result};
    use std::io::prelude::*;

    pub fn pgm_reader(path: String) -> (Vec<Vec<u32>>, usize, usize) {
    /*
    path: path of an image in file system
    normalize: `true` for intensity normalization by 1, `false`: not to change values from file
    If before normalization intensity was between 0 and 255,
    then after it intensity will be between 0 and 1.
    To do normalixation all values are divided by maximum intensity value.
    scale_factor: all values from file are divided by scale_factor
    Returns a matrix of data (pixel intensities or true disparities from file),
    number of columns and number of rows in matrix
    */
        println!("Image path: {}", path);
        let mut f = BufReader::new(File::open(path).unwrap());

        let mut num_line = String::new();
        f.read_line(&mut num_line).unwrap();
        println!("Format: '{}'", num_line.trim());
        num_line = String::new();
        f.read_line(&mut num_line).unwrap();
        let sizes: Vec<usize> = num_line
            .trim()
            .split(char::is_whitespace)
            .map(|number| number.parse().unwrap())
            .collect();
        println!("Image size: {:?}", sizes);
        let width: usize = sizes[0];
        let height: usize = sizes[1];
        num_line = String::new();
        f.read_line(&mut num_line).unwrap();
        let max_intensity: u32 = num_line.trim().parse().unwrap();
        println!("Maximum intensity: {}", max_intensity);

        let lines: Vec<String> = f.lines().map(|l| l.unwrap().trim().to_string()).collect();
        let array: Vec<i32> = lines
            .join(" ")
            .trim()
            .split(char::is_whitespace)
            .map(|number| number.parse().unwrap())
            .collect();
        assert_eq!(width * height, array.len());

        let mut matrix = vec![vec![0u32; width]; height];
        for i in 0..height {
            for j in 0..width {
                matrix[i][j] = array[i * width + j] as u32;
            }
        }
        return (matrix, width, height);
    }

    pub fn pgm_writer(matrix: &Vec<Vec<usize>>, path: String, max_intensity: usize) -> Result<()> {
    /*
    Creates or updates a .pgm file with specified `path`.
    Maximum intensity is `max_disparity`.
    Values are `depth map`
    */
        let mut file = File::create(path).expect("Unable to create file");
        writeln!(file, "P2").expect("Unable to write the magic number P2");
        writeln!(file, "{} {}", matrix[0].len(), matrix.len())
            .expect("Unable to write image size");;
        writeln!(file, "{}", max_intensity).expect("Unable to write maximum color intensity");
        for i in 0..matrix.len() {
            let string: Vec<String> = matrix[i].iter().map(|n| n.to_string()).collect();
            writeln!(file, "{}", string.join(" ")).expect("Unable to writea matrix");
        }
        Ok(())
    }
}