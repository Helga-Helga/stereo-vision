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
            .expect("Unable to write image size");
        writeln!(file, "{}", max_intensity).expect("Unable to write maximum color intensity");
        for i in 0..matrix.len() {
            let string: Vec<String> = matrix[i].iter().map(|n| n.to_string()).collect();
            writeln!(file, "{}", string.join(" ")).expect("Unable to write a matrix");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempdir::TempDir;
    use std::fs::File;
    use std::path::Path;
    use std::io::{Write};
    use super::pgm::*;

    #[test]
    fn test_pgm_reader() {
        let dir = TempDir::new("test_pgm_reader").expect("Unable to create directory");
        let file_path = dir.path().join("image.pgm");
        let file_path_copy = dir.path().join("image.pgm");

        let mut f = File::create(file_path).expect("Unable to create file");
        writeln!(f, "P2").expect("Unable to write magic number P2");
        writeln!(f, "3 2").expect("Unable to write image size");
        writeln!(f, "255").expect("Unable to write maximum color intensity");
        writeln!(f, "1 2 3").expect("Unable to write first line of image");
        writeln!(f, "4 5 6").expect("Unable to write second line of image");
        f.sync_all().expect("Unable to sync");

        let file_path_string = file_path_copy.into_os_string().into_string()
            .expect("Unable to parse file path");
        let (matrix, width, height) = pgm_reader(file_path_string);
        assert_eq!(3, width);
        assert_eq!(2, height);
        assert_eq!(1, matrix[0][0]);
        assert_eq!(2, matrix[0][1]);
        assert_eq!(3, matrix[0][2]);
        assert_eq!(4, matrix[1][0]);
        assert_eq!(5, matrix[1][1]);
        assert_eq!(6, matrix[1][2]);

        dir.close().expect("Unable to close and remove directory");
    }

    #[test]
    fn test_pgm_writer() {
        let dir = TempDir::new("test_pgm_reader").expect("Unable to create directory");
        let file_path = dir.path().join("image.pgm");
        let file_path_string = file_path.into_os_string().into_string()
            .expect("Unable to parse file path");

        let file_path_copy = dir.path().join("image.pgm");
        let file_path_string_copy = file_path_copy.into_os_string().into_string()
            .expect("Unable to parse file path");

        let matrix = [[1, 2, 3].to_vec(), [4, 5, 6].to_vec()].to_vec();
        let max_intensity: usize = 255;

        pgm_writer(&matrix, file_path_string, max_intensity).expect("Unable to write a file");

        let (matrix, width, height) = pgm_reader(file_path_string_copy);
        assert_eq!(3, width);
        assert_eq!(2, height);
        assert_eq!(1, matrix[0][0]);
        assert_eq!(2, matrix[0][1]);
        assert_eq!(3, matrix[0][2]);
        assert_eq!(4, matrix[1][0]);
        assert_eq!(5, matrix[1][1]);
        assert_eq!(6, matrix[1][2]);

        dir.close().expect("Unable to close and remove directory");
    }
}
