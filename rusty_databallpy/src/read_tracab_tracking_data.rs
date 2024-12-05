
use std::f64::NAN;
use std::fs::File;
use std::io::{self, BufReader, BufRead};
use std::collections::HashMap;
use polars::prelude::*;

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

pub fn save_tracab_dat_to_parquet(tracab_file_loc: &str) -> Result<(), io::Error> {
    let file = File::open(tracab_file_loc)?;
    let reader = BufReader::new(file);

    let mut frames_vec = Vec::new();
    let mut ball = Ball {
        x: Vec::new(),
        y: Vec::new(),
        z: Vec::new(),
        possession: Vec::new(),
        status: Vec::new(),
    };
    let mut players_hasmap: HashMap<String, Player> = HashMap::new();

    // Loop over the lines
    let mut lines_covered = 0;
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split(':').collect();
        let (frame, players_info, ball_info) = (
            parts[0].parse::<i32>().unwrap_or_else(|_| panic!("Unable to parse frame: {}", parts[0])),
            parts[1], 
            parts[2]
        );
        frames_vec.push(frame);

        let ball_info_vec: Vec<&str> = ball_info.split(",").collect();
        let (ball_x, ball_y, ball_z, ball_status, ball_possession) = (
            ball_info_vec[0].parse::<f64>().expect("Unable to parse ball_x"),
            ball_info_vec[1].parse::<f64>().expect("Unable to parse ball_y"),
            ball_info_vec[2].parse::<f64>().expect("Unable to parse ball_z"),
            ball_info_vec[3].to_string(),
            ball_info_vec[4].to_string(),
        );

        ball.x.push(ball_x);
        ball.y.push(ball_y);
        ball.z.push(ball_z);
        ball.status.push(ball_status.to_lowercase());
        ball.possession.push(match ball_possession.as_str() {
            "H" => "home".to_string(),
            "A" => "away".to_string(),
            _ => panic!("Unknown ball possession value"),
        });

        let players_info_split: Vec<&str> = players_info.split(";").collect();
            for player_info in players_info_split.iter(){
                if player_info.len() == 0 {
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

            // check if player already in hashmap
            let key = format!("{}_{}", team_side, shirt_num);

            if !players_hasmap.contains_key(&key) {
                let player = Player {
                    x: vec![NAN; lines_covered],
                    y: vec![NAN; lines_covered],
                };
                players_hasmap.insert(key.clone(), player);
            }

            let player = players_hasmap.get_mut(&key).unwrap();

            // set values
            player.x.push(x_pos);
            player.y.push(y_pos);
        }
        lines_covered += 1;
    }

    // create a polars df with all frame, ball, and players data
    let mut df = DataFrame::new(vec![
        polars::prelude::Column::Series(Series::new("frame".into(), frames_vec)),
        polars::prelude::Column::Series(Series::new("ball_x".into(), ball.x.clone())),
        polars::prelude::Column::Series(Series::new("ball_y".into(), ball.y.clone())),
        polars::prelude::Column::Series(Series::new("ball_z".into(), ball.z.clone())),
        polars::prelude::Column::Series(Series::new("ball_possession".into(), ball.possession.clone())),
        polars::prelude::Column::Series(Series::new("ball_status".into(), ball.status.clone())),
    ]).unwrap();

    // add player data to df
    for (key, player) in players_hasmap.iter() {
        let x_series = polars::prelude::Column::Series(Series::new(format!("{}_x", key).into(), &player.x));
        let y_series = polars::prelude::Column::Series(Series::new(format!("{}_y", key).into(), &player.y));

        let _ = df.with_column(x_series);
        let _ = df.with_column(y_series);
    }

    // divide all the values with column type f64 by 100 to go from cm to m
    let col_names: Vec<String> = df.get_columns()
        .iter()
        .filter(|col| col.dtype() == &DataType::Float64)
        .map(|col| col.name().to_string())
        .collect();

    for col_name in col_names {
        let col = df.column(&col_name).unwrap().f64().unwrap();
        let _ = df.with_column(col.clone().apply(|s| s.map(|v| v / 100.0)));
    }

    let file = File::create("temp_parquet.parquet")?;
    let _ = ParquetWriter::new(file).finish(&mut df);
    Ok(())
}
