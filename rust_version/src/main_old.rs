extern crate image;
extern crate line_drawing;
extern crate ndarray;
use image::{GrayImage};
use line_drawing::XiaolinWu;
use ndarray::Array2;
use std::cmp;
use std::f32::consts::PI;
use std::string;
use std::time::Instant;

fn draw_line(data: &mut Array2<u8>, start: (f32, f32), end: (f32, f32)) {
    let width = data.shape()[0] as i16;
    for ((x, y), value) in XiaolinWu::<f32, i16>::new(start, end) {
        if x < 0 || y < 0 || x >= width || y >= width {continue};
        let x = x as usize;
        let y = y as usize;
        let new_val = data[(x, y)].checked_sub((255. * value) as u8).unwrap_or(0);
        data[(x, y)] = new_val;
    }
}

fn get_loss(img: &Array2<u8>, drawing: &Array2<u8>, start: (f32, f32), end: (f32, f32)) -> (i32, i32) {
    let OVER_RATIO = 0f32;
    let width = img.shape()[0] as i16;
    let mut loss: i32 = 0;
    let mut length: i32 = 0;
    for ((x, y), value) in XiaolinWu::<f32, i16>::new(start, end) {
        if x < 0 || y < 0 || x >= width || y >= width {continue};
        length += 1;
        let x = x as usize;
        let y = y as usize;
        let value_u8 = (value * 255.) as u8;
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

fn peg_pos(n: u32, width: u32) -> Vec<(f32, f32)> {
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

fn manhattan(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn find_min(img: &Array2<u8>, drawing: &Array2<u8>, peg_pos: &Vec<(f32, f32)>) -> (usize, usize) {
    let mut min = i32::MAX;
    let mut argmin = (0usize, 1usize);
    // Call the function you want to time
    for i in 0..179 {
        for j in i+1..180 {
            let ret = get_loss(&img, &drawing, peg_pos[i], peg_pos[j]);
            let mut loss = ret.0;
            let length = ret.1;
            loss /= length;
            // println!("({:03}, {:03}) -> {}", i, j, loss);
            if loss < min {
                min = loss;
                argmin = (i, j)
            }
        }
    }
    // println!("___min: {:?}", min);
    argmin
}

fn main() {
    let width: usize = 1024;
    let mut drawing: Array2<u8> = Array2::zeros((width, width));
    drawing.fill(255);

    let preprocessed: Array2<u8> = preprocess("dog.jpeg".to_owned(), width as u32, width as u32);
    let peg_pos = peg_pos(180, width as u32);

    
    let mut start = Instant::now();
    for i in 0..1000 {
        let min = find_min(&preprocessed, &drawing, &peg_pos);
        // println!("argmin: {:?}", min);
        let poz = &peg_pos[min.0];
        draw_line(&mut drawing, peg_pos[min.0], peg_pos[min.1]);
        if (i + 1) % 20 == 0 {
            let duration = start.elapsed();
            let raw = drawing.clone().into_raw_vec();
            let img = GrayImage::from_raw(width as u32, width as u32, raw).expect("Uh oh");
            match img.save(format!("images_1/out_{:03}.png", i + 1)) {
                Ok(_) => {
                    println!("Saved images_1/out_{:03}.png, duration: {:?}", i + 1, duration);
                },
                Err(_) => {
                    println!("Error saving images_1/out_{:03}.png", i + 1);
                },
            }
            start = Instant::now();
        }
    }

    let raw = drawing.into_raw_vec();
    let img = GrayImage::from_raw(width as u32, width as u32, raw).expect("Uh oh");
    img.save("out.png");
}
