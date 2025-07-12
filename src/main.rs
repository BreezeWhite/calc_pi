use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::time::Instant;

use calc_pi::{Actions, Cli, algos, get_iters_and_prec};
use clap::Parser;
use rug::float::Round;

#[allow(dead_code)]
fn cmp_pi(pi: String) {
    let file = match File::open("PI_10M.txt") {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open PI_10M.txt: {}", e);
            return;
        }
    };
    let reader = BufReader::new(file);
    let mut file_bytes = reader.bytes();
    let mut correct = 0;

    for c in pi.chars() {
        // Skip non-digit and non-dot chars in file
        let mut file_c = None;
        while let Some(Ok(b)) = file_bytes.next() {
            let ch = b as char;
            if ch.is_ascii_digit() || ch == '.' {
                file_c = Some(ch);
                break;
            }
        }
        match file_c {
            Some(fc) if fc == c => correct += 1,
            Some(_) | None => break,
        }
    }
    println!("Matched digits: {}", correct);
}

fn main() {
    let arg = Cli::parse();

    let decimal_prec = arg.prec;
    let (iters, prec) = get_iters_and_prec(arg.action.clone(), decimal_prec);
    // println!("{iters} {prec}");

    let pi_func = match arg.action {
        Actions::Leibniz => algos::pi_leibniz,
        Actions::BBP => algos::pi_bailey_borwein_plouffe_v2,
        Actions::BN => algos::pi_borwein_novin,
        Actions::BS => algos::pi_brent_salamin,
        Actions::GL => algos::pi_gauss_legendre,
        Actions::Chu => algos::pi_chudnovsky,
        Actions::CB => algos::pi_chudnovsky_binary_splitting,
        Actions::CBP => algos::pi_chudnovsky_binary_splitting_parallel,
        Actions::AG => algos::pi_arctan_gauss,
        Actions::SPG => algos::pi_spigot_gosper,
        Actions::Newton => algos::pi_newton_9th,
    };

    let start = Instant::now();
    let pi = pi_func(0, iters, prec);
    let dura = start.elapsed();

    let pi = format!(
        "{}",
        pi.to_string_radix_round(10, Some(decimal_prec), Round::Down)
    );

    // cmp_pi(pi.clone());

    if let Some(path) = arg.output_to {
        let mut file = match File::create(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create {}: {}", path, e);
                return;
            }
        };
        if let Err(e) = file.write_all(pi.as_bytes()) {
            eprintln!("Failed to write to {}: {}", path, e);
        }
    } else {
        println!("{pi}");
    }

    if arg.measure_time {
        println!("Elapsed: {dura:?}");
    }
}
