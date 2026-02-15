fn main() {
  let mut data = Array2::<u8>::zeros((10, 10));
  let pegs = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
  let peg_pos = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
  // let preprocessed = ...; // your preprocessed data
  // let drawing = ...; // your drawing data

  let start = Instant::now();
  let min = find_min(&preprocessed, &drawing, peg_pos);

  let duration = start.elapsed();
  println!("Minimum: {:?}", min);
  println!("Duration: {:?}", duration);

  // Use references to avoid moving peg_pos
  let poz = &peg_pos[min.0];

  // Dereference the references when passing to the function
  draw_line(&mut data, *poz, peg_pos[min.1]);
}

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
