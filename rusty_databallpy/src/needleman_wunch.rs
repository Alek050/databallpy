use ndarray::{s, Array, Array1, Array2};
use std::collections::HashMap;



pub fn needleman_wunch(
    sim_mat: Array2<f32>,
    gap_event: f32,
    gap_frame: f32
) -> HashMap<i32, i32> {
    let n_frames: usize = sim_mat.shape()[0];
    let n_events: usize = sim_mat.shape()[1];

    // function matrix
    let mut function_matrix: Array2<f32> = Array2::zeros((n_frames + 1, n_events + 1));
    let gap_event_penalties: Array1<f32> = Array::linspace(0.0, (n_events as f32) * gap_event , n_events + 1);
    let gap_frames_penalties: Array1<f32> = Array::linspace(0.0, (n_frames as f32) * gap_frame, n_frames + 1);
    function_matrix.slice_mut(s![..,0]).assign(&gap_frames_penalties);
    function_matrix.slice_mut(s![0,..]).assign(&gap_event_penalties);

    // pointer matrix
    let mut pointer_matrix: Array2<i16> = Array2::zeros((n_frames + 1, n_events + 1));
    pointer_matrix.slice_mut(s![..,0]).assign(&Array::from_elem(n_frames + 1, 3));
    pointer_matrix.slice_mut(s![0,..]).assign(&Array::from_elem(n_events + 1, 4));

    // fill pointer and function matrix
    for i in 0..n_frames {
        for j in 0..n_events {
            let val_0 = function_matrix[[i, j]] + sim_mat[[i, j]];
            let val_1 = function_matrix[[i, j + 1]] + gap_frame;
            let val_2 = function_matrix[[i + 1, j]] + gap_event;
    
            let (t_max, value) = if val_0 >= val_1 && val_0 >= val_2 {
                // val 0 is highest, thus we have a match
                (val_0, pointer_matrix[[i + 1, j + 1]] + 2)
            } else if val_1 >= val_2 {
                // val 1 is highest, thus frame unassigned
                (val_1, pointer_matrix[[i + 1, j + 1]] + 3)
            } else {
                // val 2 is highest, thus we have an unassigned event
                (val_2, pointer_matrix[[i + 1, j + 1]] + 4)
            };
    
            pointer_matrix.slice_mut(s![i + 1, j + 1]).fill(value);
            function_matrix.slice_mut(s![i + 1, j + 1]).fill(t_max);
        }
    }

    // trace through the optimal alignment
    let mut i: usize = n_frames;
    let mut j: usize = n_events;
    let mut idx: i32 = 0;
    let mut frames: Array1<i32> = Array::zeros(n_frames);
    let mut events: Array1<i32> = Array::zeros(n_frames);
    while i > 0 || j > 0 {
        let value: i16 = pointer_matrix[[i, j]];
        match value {
            2 | 5 | 6 | 9 => {
                frames.slice_mut(s![idx]).fill(i.try_into().unwrap());
                events.slice_mut(s![idx]).fill(j.try_into().unwrap());
                j -= 1;
            },
            3 | 7 => {
                frames.slice_mut(s![idx]).fill(i.try_into().unwrap());
                events.slice_mut(s![idx]).fill(0);
            },
            4 => panic!("An event was left unassigned, check your gap penalty values"),
            _ => panic!("Unexpected value in pointer matrix at P[{}, {}]: {}", i, j, value),
        }
        i -= 1;
        idx += 1;
    }

    let idx_events: Vec<i32> = events
        .iter()
        .enumerate()
        .filter(|&(_, &i)| i > 0)
        .map(|(idx, _)| idx as i32)
        .collect();

    let mut event_frame_dict: HashMap<i32, i32> = HashMap::new();
    for i in idx_events.iter() {
        event_frame_dict.insert(events[*i as usize] - 1, frames[*i as usize] - 1);
    }
    event_frame_dict
}
