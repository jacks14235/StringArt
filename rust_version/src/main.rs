extern crate image;
extern crate line_drawing;
extern crate ndarray;
use image::{GrayImage};
use line_drawing::XiaolinWu;
use ndarray::{Array, Array2};
use std::cmp;
use std::collections::{BinaryHeap, HashMap};
use std::f32::consts::PI;
use std::time::Instant;

// returns hashmap where keys are starting and ending peg numbers and value is Vec<(affected pixel x, affected pixel y, opacity)>
fn precalc(pegs: &Vec<(f32, f32)>) -> HashMap<(usize, usize), Vec<((i16, i16), f32)>> {
    let mut map = HashMap::new();
    let n_pegs = pegs.len();
    for i in 0..n_pegs - 1 {
        for j in i+1..n_pegs {
            map.insert((i, j), XiaolinWu::<f32, i16>::new(pegs[i], pegs[j]).collect());
        }
    }
    map
}

fn draw_line(data: &mut Array2<u8>, start: (f32, f32), end: (f32, f32)) {
    let width = data.shape()[0] as i16;
    for ((x, y), value) in XiaolinWu::<f32, i16>::new(start, end) {
        if x < 0 || y < 0 || x >= width || y >= width {continue};
        let x = x as usize;
        let y = y as usize;
        let new_val = data[(x, y)].checked_sub((64. * value) as u8).unwrap_or(0);
        data[(x, y)] = new_val;
    }
}

fn get_loss(img: &Array2<u8>, drawing: &Array2<u8>, precalc: &HashMap<(usize, usize), Vec<((i16, i16), f32)>>, start: usize, end: usize) -> (i32, i32) {
    let OVER_RATIO = 0.4f32;
    let width = img.shape()[0] as i16;
    let mut loss: i32 = 0;
    let mut length: i32 = 0;
    let vec: &Vec<((i16, i16), f32)> = precalc.get(&(start, end)).expect(&format!("{}, {} was not present in the hashmap", start, end));
    for point in vec {
        let ((x, y), value) = *point;
        if x < 0i16 || y < 0i16 || x >= width || y >= width {continue};
        length += 1;
        let x = x as usize;
        let y = y as usize;
        let value_u8 = (value * 64.) as u8;
        let target = img[(x,y)];
        let curr = drawing[(x,y)];
        let next = (curr).checked_sub(value_u8).unwrap_or(0);
        
        if next < target {
            // penalize by the value we increased past target, 
            // but only as much as we are adding on this step
            loss += ((cmp::min(target - next, curr -  next) as f32) * OVER_RATIO) as i32;
        } else {
            loss -= (curr - next) as i32;
        }
    }
    // println!("loss: {}", loss);
    (loss, length)
}

fn preprocess(path: String, width: u32, height: u32) -> Array2<u8> {
    let img = image::open(path).expect("Failed to open image");

    // Resize the image
    let resized_img = img.resize(width, height, image::imageops::FilterType::CatmullRom);

    // Convert the image to grayscale
    let grayscale_img = resized_img.to_luma8();

    // Convert the grayscale image to a u8 array
    let raw_data = grayscale_img.into_raw();

    let array = Array2::from_shape_vec((height as usize, width as usize), raw_data).expect("Failed to create Array2");
    array
}

fn circle_pegs(n: u32, width: u32) -> Vec<(f32, f32)> {
    let mut coords = Vec::new();
    let half_width = (width / 2) as f32;
    for i in 0..n {
        let a = 2. * PI * (i as f32) / (n as f32);
        coords.push((
            half_width + half_width * (f32::cos(a)),
            half_width + half_width * (f32::sin(a))
        ));
    }
    coords
}

fn square_pegs(n: u32, width: u32) -> Vec<(f32, f32)> {
    if n % 4 != 0 {
        panic!("Number of pegs must be divisible by 4 for square image");
    }
    let mut coords = Vec::new();
    let dx = width as f32 / ((n / 4) as f32);
    // top
    for i in 0..n/4 {
        coords.push(((i as f32) * dx, 0.));
    }
    // right
    for i in 0..n/4 {
        coords.push(((width - 1) as f32, (i as f32) * dx));
    }
    // bottom
    for i in 0..n/4 {
        coords.push((width as f32 - dx * i as f32, (width - 1) as f32));
    }
    // left
    for i in 0..n/4 {
        coords.push((0., width as f32 - dx * i as f32));
    }
    coords
}

fn find_min(img: &Array2<u8>, drawing: &Array2<u8>, precalc: &HashMap<(usize, usize), Vec<((i16, i16), f32)>>, n_pegs: usize) -> Vec<(usize, usize)> {
    let k = 20; // number to keep saved
    let mut heap = BinaryHeap::with_capacity(k);
    
    // Call the function you want to time
    for i in 0..n_pegs - 1 {
        for j in i+1..n_pegs {
            let ret = get_loss(&img, &drawing, &precalc, i, j);
            let mut loss = ret.0;
            let length = ret.1;
            loss /= length;
            // println!("({:03}, {:03}) -> {}", i, j, loss);
            if heap.len() < k {
                heap.push((loss, (i, j)));
            } else if let Some(&largest) = heap.peek() {
                if loss < largest.0 {
                    heap.pop();
                    heap.push((loss, (i, j)));
                }
            }
        }
    }
    heap.into_iter().map(|a| a.1).collect()
}

fn main() {
    let width: usize = 1024;
    let n_pegs: usize = 180;
    let mut drawing: Array2<u8> = Array2::zeros((width, width));
    drawing.fill(255);

    let preprocessed: Array2<u8> = preprocess("lion.jpeg".to_owned(), width as u32, width as u32);
    let peg_pos = square_pegs(n_pegs as u32, width as u32);
    let precalc = precalc(&peg_pos);

    
    let mut start = Instant::now();
    for i in 0..1000 {
        let mins = find_min(&preprocessed, &drawing, &precalc, n_pegs);
        // println!("argmin: {:?}", min);
        for min in mins {
            draw_line(&mut drawing, peg_pos[min.0], peg_pos[min.1]);
        }
        if (i + 1) % 4 == 0 {
            let duration = start.elapsed();
            let raw = drawing.clone().into_raw_vec();
            let img = GrayImage::from_raw(width as u32, width as u32, raw).expect("Uh oh");
            let path = format!("images_1/out_loss_{:03}.png", i + 1);
            match img.save(&path) {
                Ok(_) => {
                    println!("Saved {}, duration: {:?}", &path, duration);
                },
                Err(_) => {
                    println!("Error saving {}", &path);
                },
            }
            start = Instant::now()
        }
    }
    
    let raw = drawing.into_raw_vec();
    let img = GrayImage::from_raw(width as u32, width as u32, raw).expect("Uh oh");
    img.save("out.png");
}
