use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::thread;
use std::time::Instant;

use clap::{Parser, Subcommand};
use rayon::prelude::*;
use rug::{
    Assign, Float, Integer,
    float::Round,
    ops::{AddFrom, DivFrom, MulFrom, Pow, SubFrom},
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    action: Actions,

    /// Precision of Pi to calculate in digits.
    #[arg(short, long, default_value_t = 1000)]
    prec: usize,

    /// Measure and show the runtime.
    #[arg(long)]
    measure_time: bool,

    /// Path of file to output ad store the calculated Pi digits.
    #[arg(long)]
    output_to: Option<String>,
}

#[derive(Subcommand, Clone)]
enum Actions {
    /// Leibniz formula
    Leibniz,
    /// Bailey–Borwein–Plouffe formula
    BBP,
    /// Borwein's algorithm nonic (9th) convergence version
    BN,
    /// Brent–Salamin algorithm
    BS,
    /// Gauss–Legendre algorithm
    GL,
    /// Chudnovsky algorithm
    Chu,
    /// Chudnovsky algorithm with binary splitting
    CB,
    /// Chudnovsky algorithm with binary splitting and multi-thread
    CBP,
    /// Machin-like formulas (arctan)
    AG,
}

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

fn decide_iterations(act: Actions, decimal_prec: usize) -> u64 {
    match act {
        Actions::Leibniz => {
            if decimal_prec > 19 {
                panic!("Overflow! Please use a smaller precision (<= 19) for Leibniz formula.");
            }
            10_u64.pow(decimal_prec as u32) / 2
        }
        Actions::BBP => decimal_prec as u64,
        Actions::BN => {
            let log9 = 9_f64.log10();
            let p = decimal_prec as f64;
            let iters = p.log10() / log9;
            iters.ceil() as u64 + 1
        }
        Actions::BS => {
            let p = decimal_prec as f64;
            let iters = p.log2();
            iters.ceil() as u64 + 1
        }
        Actions::GL => {
            let p = decimal_prec as f64;
            let iters = p.log2();
            iters.ceil() as u64 + 1
        }
        Actions::Chu => decimal_prec as u64 / 14 + 1,
        Actions::CB => decimal_prec as u64 / 14 + 1,
        Actions::CBP => decimal_prec as u64 / 14 + 1,
        Actions::AG => 1,
    }
}

fn pi_leibniz(start_idx: u64, end_idx: u64, prec: u32) -> Float {
    // https://en.wikipedia.org/wiki/Leibniz_formula_for_π
    let mut nume = if start_idx % 2 == 0 {
        Integer::from(1)
    } else {
        Integer::from(-1)
    };

    let mut deno = Integer::from(0);
    let mut s1 = Float::with_val(prec, 0);
    let mut sum = Float::with_val(prec, 0);
    for i in start_idx..end_idx {
        deno.assign(i * 2);
        deno += 1;
        s1.assign(&nume);
        s1 /= &deno;
        sum += &s1;
        nume *= -1;
    }

    sum * 4
}

#[allow(dead_code)]
fn pi_bailey_borwein_plouffe(start_idx: u64, end_idx: u64, prec: u32) -> Float {
    // https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula
    let mut pi = Float::with_val(prec, 0);
    let mut pi1 = Float::with_val(prec, 0);
    let mut deno: Float = Float::with_val(prec, 1);
    let mut comm: Float = Float::with_val(prec, 0);

    let mut a = Float::with_val(prec, 1);
    let mut b = Float::with_val(prec, 1);
    let mut c = Float::with_val(prec, 1);
    let mut d = Float::with_val(prec, 1);

    for _ in start_idx..end_idx {
        a.assign(&comm);
        a += 1;
        a.div_from(4);

        b.assign(&comm);
        b += 4;
        b.div_from(2);

        c.assign(&comm);
        c += 5;
        c.div_from(1);

        d.assign(&comm);
        d += 6;
        d.div_from(1);

        pi1.assign(&a);
        pi1 -= &b;
        pi1 -= &c;
        pi1 -= &d;
        pi1 /= &deno;
        pi += &pi1;

        comm += 8;
        deno *= 16;
    }

    pi
}

fn pi_bailey_borwein_plouffe_v2(start_idx: u64, end_idx: u64, prec: u32) -> Float {
    // https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula
    let mut pi = Float::with_val(prec, 0);
    let mut pi1 = Float::with_val(prec, 0);

    let mut k2 = Integer::from(0);
    let mut k3 = Integer::from(0);
    let mut k4 = Integer::from(0);
    let mut deno = Integer::from(0);
    let mut deno16 = Integer::from(1);
    let mut nomi = Integer::from(0);

    for i in start_idx..end_idx {
        k2.assign(i * i);
        k3.assign(&k2 * i);
        k4.assign(&k3 * i);

        deno.assign(15);
        deno += 194 * i;
        deno += &k2 * 712;
        deno += &k3 * 1024;
        deno += &k4 * 512;
        deno *= &deno16;
        deno16 *= 16;

        nomi.assign(47);
        nomi += 151 * i;
        nomi += &k2 * 120;

        pi1.assign(&nomi);
        pi1 /= &deno;
        pi += &pi1;
    }

    pi
}

fn pi_borwein_novin(start_idx: u64, end_idx: u64, prec: u32) -> Float {
    // Borwein's algorithm using Ramanujan–Sato series.
    // This implements the nonic convergence of Borwein's algorithm.
    // https://en.wikipedia.org/wiki/Borwein%27s_algorithm#Nonic_convergence
    let one_third: Float = Float::with_val(prec, 1) / 3;

    let mut a: Float = one_third.clone();
    let mut r: Float = (Float::with_val(prec, 3).sqrt() - 1) / 2;
    let mut s: Float = 1 - r.clone().pow(3);
    s = s.pow(one_third.clone());

    let mut t = Float::with_val(prec, 0);

    let mut u = Float::with_val(prec, 0);
    let mut ur_square = Float::with_val(prec, &r).square();
    let mut ur = Float::with_val(prec, 0);

    let mut v = Float::with_val(prec, 0);
    let mut vt_square = Float::with_val(prec, 0);
    let mut vu_square = Float::with_val(prec, 0);

    let mut w = Float::with_val(prec, 0);
    let mut ws_square = Float::with_val(prec, 0);

    let mut aw = Float::with_val(prec, 0);
    let mut aw_fac = Float::with_val(prec, 0);

    let mut s_deno = Float::with_val(prec, 0);

    for i in start_idx..end_idx {
        // Construct t
        t.assign(&r);
        t = t * 2 + 1;

        // Construct u
        ur_square.assign(&r);
        ur_square = ur_square.square();
        ur.assign(&r);
        ur.add_from(&ur_square);
        ur += 1;
        u.assign(&r);
        u.mul_from(9);
        u.mul_from(&ur);
        u = u.pow(&one_third);

        // Construct v
        v.assign(&t);
        v.mul_from(&u);
        vt_square.assign(&t);
        vt_square = vt_square.square();
        vu_square.assign(&u);
        vu_square = vu_square.square();
        v.add_from(&vt_square);
        v.add_from(&vu_square);

        // Construct w
        ws_square.assign(&s);
        ws_square = ws_square.square();
        w.assign(&s);
        w.add_from(&ws_square);
        w = 27 * (1 + w);
        w.div_from(&v);
        w = w.pow(-1);

        // Update a
        aw.assign(&w);
        aw = 1 - aw;
        aw_fac.assign(3);
        aw_fac = aw_fac.pow(2 * i as i64 - 1);
        aw.mul_from(&aw_fac);
        a.mul_from(&w);
        a.add_from(&aw);

        // Update s
        s.assign(&r);
        s = 1 - s;
        s = s.pow(3);
        s_deno.assign(&u);
        s_deno *= 2;
        s_deno.add_from(&t);
        s_deno.mul_from(&v);
        s.div_from(&s_deno);
        s = s.pow(-1);

        // Update r
        r.assign(&s);
        r = r.pow(3);
        r = 1 - r;
        r = r.pow(&one_third);
    }
    1 / a
}

fn pi_brent_salamin(start_idx: u64, end_idx: u64, prec: u32) -> Float {
    // Brent–Salamin algorithm
    let mut a = Float::with_val(prec, 1);
    let mut a_next = Float::with_val(prec, 1);
    let mut b: Float = Float::with_val(prec, 2).sqrt().pow(-1);
    let mut b_next: Float = Float::with_val(prec, &b);
    let mut c = Float::with_val(prec, 0.5);
    let mut cp2 = Float::with_val(prec, 2);
    let mut cp = Float::with_val(prec, 0);

    for _ in start_idx..end_idx {
        a_next.assign(&a);
        a_next += &b;
        a_next /= 2;
        b_next *= &a;
        b_next = b_next.sqrt();
        cp.assign(&a_next);
        cp -= &b;
        cp = cp.square();
        cp *= &cp2;
        cp2 *= 2;
        c -= &cp;

        a.assign(&a_next);
        b.assign(&b_next);
    }

    2 * a.square() / c
}

fn pi_gauss_legendre(start_idx: u64, end_idx: u64, prec: u32) -> Float {
    // Gauss–Legendre algorithm. Similar to Brent–Salamin algorithm.
    // https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_algorithm
    let mut a = Float::with_val(prec, 1);
    let mut a_next = Float::with_val(prec, 1);
    let mut b: Float = Float::with_val(prec, 2).sqrt().pow(-1);
    let mut t = Float::with_val(prec, 1) / 4;
    let mut tp = Float::with_val(prec, 0);
    let mut p = Float::with_val(prec, 1);

    for _ in start_idx..end_idx {
        a_next.assign(&a);
        a_next.add_from(&b);
        a_next /= 2;

        b *= &a;
        b = b.sqrt();

        tp.assign(&a);
        tp.sub_from(&a_next);
        tp = tp.square();
        tp *= &p;
        t -= &tp;

        p *= 2;
        a.assign(&a_next);
    }

    let pi = (a + b).square() / (4 * t);
    pi
}

fn pi_chudnovsky(start_idx: u64, end_idx: u64, prec: u32) -> Float {
    // Chudnovsky algorithm
    // This algorithm is recursive, thus cannot spread the computation across multiple threads.
    // https://en.wikipedia.org/wiki/Chudnovsky_algorithm

    let prec = prec;
    let mut a = Float::with_val(prec, 1);
    let mut a1 = Integer::from(1);
    let mut a2 = Integer::from(1);
    let mut a3 = Integer::from(1);
    let mut a_deno = Integer::from(1);
    let a_deno_const = 10939058860032000_u64;
    let mut a_sum = Float::with_val(prec, 1);

    let mut b_sum = Float::with_val(prec, 0);
    let mut ba = Float::with_val(prec, 0);
    let mut n = Integer::from(0);
    for i in start_idx..end_idx {
        n.assign(i + 1);

        a1.assign(&n);
        a1 = a1 * 6 - 5;
        a2.assign(&n);
        a2 = a2 * 2 - 1;
        a3.assign(&n);
        a3 = a3 * 6 - 1;
        a1 *= &a2;
        a1 *= &a3;

        a_deno.assign(&n);
        a_deno = a_deno.pow(3);
        a_deno *= &a_deno_const;
        a *= &a1;
        a /= &a_deno;
        a *= -1;
        a_sum += &a;

        ba.assign(&a);
        ba *= &n;
        b_sum += &ba;
    }

    let mut pi = 426880 * Float::with_val(prec, 10005).sqrt();
    pi /= 13591409 * a_sum + 545140134 * b_sum;
    pi
}

fn pi_chu_binary_sub(
    a: u64,
    b: u64,
    prec: u32,
    mut pab: Integer,
    mut qab: Integer,
    mut tab: Integer,
) -> (Integer, Integer, Integer) {
    if b - a == 1 {
        if a > 0 {
            pab.assign(6 * a - 5);
            pab *= 2 * a - 1;
            pab *= 6 * a - 1;
            qab.assign(a);
            qab = qab.pow(3);
            qab *= 10939058860032000_u64;
        }
        tab.assign(13591409 + 545140134 * a);
        tab *= &pab;
        if a % 2 == 1 {
            tab *= -1;
        }
    } else {
        let mut pmb = Integer::from(1);
        let mut qmb = Integer::from(1);
        let mut tmb = Integer::from(0);

        let m = (a + b) / 2;
        (pab, qab, tab) = pi_chu_binary_sub(a, m, prec, pab, qab, tab);
        (pmb, qmb, tmb) = pi_chu_binary_sub(m, b, prec, pmb, qmb, tmb);
        tmb *= &pab;
        tab *= &qmb;
        tab += tmb;
        pab *= &pmb;
        qab *= &qmb;
    }

    (pab, qab, tab)
}

fn pi_chudnovsky_binary_splitting(start_idx: u64, end_idx: u64, prec: u32) -> Float {
    // Chudnovsky binary splitting algorithm
    // https://www.craig-wood.com/nick/articles/pi-chudnovsky/

    let pab = Integer::from(1);
    let qab = Integer::from(1);
    let tab = Integer::from(0);

    let (_, q, t) = pi_chu_binary_sub(start_idx, end_idx, prec, pab, qab, tab);

    let mut pi = Float::with_val(prec, q);
    pi /= t;
    pi *= 426880;
    pi *= Float::with_val(prec, 10005).sqrt();
    pi
}

fn pi_chudnovsky_binary_splitting_parallel(start_idx: u64, end_idx: u64, prec: u32) -> Float {
    // Use half of the CPU cores to compute in parallel.
    let thread_cnt = thread::available_parallelism().map(|n| n.get()).unwrap() / 2;
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_cnt)
        .build_global()
        .unwrap();
    let step_size: u64 = (end_idx - start_idx) / thread_cnt as u64;

    let results: Vec<(Integer, Integer, Integer)> = (0..thread_cnt)
        .into_par_iter()
        .map(|i| {
            let pab = Integer::from(1);
            let qab = Integer::from(1);
            let tab = Integer::from(0);
            let a = i as u64 * step_size;
            let b = a + step_size;
            pi_chu_binary_sub(a, b, prec, pab, qab, tab)
        })
        .collect();

    let mut pab = Integer::from(1);
    let mut qab = Integer::from(1);
    let mut tab = Integer::from(0);

    for (pmb, qmb, mut tmb) in results {
        let _: Vec<_> = vec![(&mut qab, &qmb), (&mut tab, &qmb), (&mut tmb, &pab)]
            .into_par_iter()
            .map(|(a, b)| a.mul_from(b))
            .collect();

        pab *= pmb;
        tab += tmb;
    }

    let mut pi = Float::with_val(prec, qab);
    pi /= tab;
    pi *= 426880;
    pi *= Float::with_val(prec, 10005).sqrt();
    pi
}

fn pi_arctan_gauss(_start_idx: u64, _end_idx: u64, prec: u32) -> Float {
    // Arctan Gauss algorithm
    // https://www.craig-wood.com/nick/articles/pi-machin/
    // https://en.wikipedia.org/wiki/Machin-like_formula#Two-term_formulas
    let c18: Float = 1 / Float::with_val(prec, 18);
    let c57: Float = 1 / Float::with_val(prec, 57);
    let c239: Float = 1 / Float::with_val(prec, 239);
    let pi = 4 * (12 * c18.atan() + 8 * c57.atan() - 5 * c239.atan());
    pi
}

fn main() {
    let arg = Cli::parse();

    let decimal_prec = arg.prec;
    let prec = (decimal_prec as f64 * 3.3219280948874).round() as u32 + 10;
    let iters = decide_iterations(arg.action.clone(), decimal_prec);
    // println!("{iters} {prec}");

    let pi_func = match arg.action {
        Actions::Leibniz => pi_leibniz,
        Actions::BBP => pi_bailey_borwein_plouffe_v2,
        Actions::BN => pi_borwein_novin,
        Actions::BS => pi_brent_salamin,
        Actions::GL => pi_gauss_legendre,
        Actions::Chu => pi_chudnovsky,
        Actions::CB => pi_chudnovsky_binary_splitting,
        Actions::CBP => pi_chudnovsky_binary_splitting_parallel,
        Actions::AG => pi_arctan_gauss,
    };

    let start = Instant::now();
    let pi = pi_func(0, iters, prec);
    let dura = start.elapsed();

    let pi = format!(
        "{}",
        pi.to_string_radix_round(10, Some(decimal_prec), Round::Down)
    );

    cmp_pi(pi.clone());

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
