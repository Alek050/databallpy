use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufRead};
use std::f64::NAN;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;

struct Player {
    x: Vec<f64>,
    y: Vec<f64>,
}

struct Ball {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    possession: Vec<String>,
    status: Vec<String>,
}

pub fn read_tracab_txt_data(tracab_file_loc: &str, verbose:bool) -> Result<(
    HashMap<String, Vec<f64>>,
    HashMap<String, Vec<String>>,
    Vec<i32>,
), io::Error> {
    let file = File::open(tracab_file_loc)?;
    let reader = BufReader::new(file);
    let total_lines = reader.lines().count();
    let mut frames_vec = Vec::new();
    let mut ball = Ball {
        x: Vec::with_capacity(total_lines),
        y: Vec::with_capacity(total_lines),
        z: Vec::with_capacity(total_lines),
        possession: Vec::with_capacity(total_lines),
        status: Vec::with_capacity(total_lines),
    };
    let mut players_hasmap: HashMap<String, Player> = HashMap::new();
    
    let pb = if verbose {
        let pb = ProgressBar::new(total_lines as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
            .progress_chars("#>-"));
        Some(pb)
    } else {
        None
    };

    // Loop over the lines
    let file = File::open(tracab_file_loc)?;
    let reader = BufReader::new(file);
    let mut lines_covered = 0;
    for line in reader.lines() {
        if let Some(ref pb) = pb {
            pb.inc(1);
        }
        let line = line?;
        let parts: Vec<&str> = line.split(':').collect();
        let (frame, players_info, ball_info) = (
            parts[0].parse::<i32>().unwrap_or_else(|_| panic!("Unable to parse frame: {}", parts[0])),
            parts[1], 
            parts[2]
        );
        frames_vec.push(frame);

        let ball_info_vec: Vec<&str> = ball_info.split(",").collect();
        let (ball_x, ball_y, ball_z, ball_possession, ball_status) = (
            ball_info_vec[0].parse::<f64>().expect("Unable to parse ball_x"),
            ball_info_vec[1].parse::<f64>().expect("Unable to parse ball_y"),
            ball_info_vec[2].parse::<f64>().expect("Unable to parse ball_z"),
            ball_info_vec[4].to_string(),
            ball_info_vec[5].to_string(),
        );

        ball.x.push(ball_x / 100.);
        ball.y.push(ball_y / 100.);
        ball.z.push(ball_z / 100.);
        ball.status.push(ball_status.trim_end_matches(";").to_lowercase());
        ball.possession.push(match ball_possession.as_str() {
            "H" => "home".to_string(),
            "A" => "away".to_string(),
            _ => panic!("Unknown ball possession value"),
        });

        let mut players_to_update: Vec<String> = players_hasmap.keys().cloned().collect();
        for player_info in players_info.split(";") {
            if player_info.is_empty() {
                continue;
            }
            let player_vec: Vec<&str> = player_info.split(",").collect();
            let (team_id, shirt_num, x_pos, y_pos) = (
                player_vec[0], 
                player_vec[2],
                player_vec[3].parse::<f64>().expect("Could not parse x_pos"),
                player_vec[4].parse::<f64>().expect("Could not parse y_pos"),
            );
            let team_side = match team_id {
                "0" => "away",
                "1" => "home",
                _ => "referee",
            };

            if team_side == "referee" {
                continue;
            }

            let key = format!("{}_{}", team_side, shirt_num);
            let player = players_hasmap.entry(key.clone()).or_insert_with(|| {
                let mut x = Vec::with_capacity(total_lines);
                let mut y = Vec::with_capacity(total_lines);
                x.resize(lines_covered, NAN);
                y.resize(lines_covered, NAN);
                Player { x, y }
            });

            // set values
            player.x.push(x_pos / 100.);
            player.y.push(y_pos / 100.);
            
            // remove player from players_to_update
            players_to_update.retain(|x| x != &key);
        }

        // add nan for all players that needed updating but did not have a value
        for player_to_update in players_to_update.iter() {
            let player = players_hasmap.get_mut(player_to_update).unwrap();
            player.x.push(NAN);
            player.y.push(NAN);
        }

        lines_covered += 1;
    }

    let mut floats_dict: HashMap<String, Vec<f64>> = HashMap::from([
        ("ball_x".to_owned(), ball.x),
        ("ball_y".to_owned(), ball.y),
        ("ball_z".to_owned(), ball.z),
    ]);
    for player_key in players_hasmap.keys() {
        let new_key_x = format!("{}_x", player_key);
        let new_key_y = format!("{}_y", player_key);
        let player = players_hasmap.get(player_key).unwrap();
        floats_dict.insert(new_key_x, player.x.clone());
        floats_dict.insert(new_key_y, player.y.clone());
    }

    let strings_dict: HashMap<String, Vec<String>> = HashMap::from([
        ("ball_possession".to_owned(), ball.possession),
        ("ball_status".to_owned(), ball.status),
    ]);

    // return floats, strings, and frames
    Ok((floats_dict, strings_dict, frames_vec))
}
