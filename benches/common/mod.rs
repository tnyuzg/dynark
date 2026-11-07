//! Helpers shared by the benchmarks: command-line options, recording runs,
//! and writing results.
//!
//! Every run of a benchmark writes two files to `results/` (or `--out-dir`),
//! named after the benchmark and the start time:
//!
//! - `<bench>-<time>.csv`: one row per measured run. This is the data that
//!   tables and figures are built from.
//! - `<bench>-<time>.md`: the command, commit and machine, and a summary
//!   table (the same one printed to the terminal).
//!
//! Every benchmark accepts `--log-n`, `--repeat`, `--threads` and
//! `--out-dir`; see `benches/README.md` for the rest.
#![allow(dead_code)]

use std::{
    collections::HashMap,
    fmt::Write as _,
    fs,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Whether an extra option takes a value or is a switch.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Arg {
    /// `--name value`
    Value,
    /// `--name`
    Switch,
}

/// Command-line options.
pub struct Options {
    /// The benchmark's name.
    pub name: &'static str,
    /// Domain sizes, as powers of two.
    pub log_n: Vec<usize>,
    /// Runs per measurement.
    pub repeat: usize,
    /// Size of the thread pool.
    pub threads: usize,
    /// Where to write the results.
    pub out_dir: PathBuf,
    extra: HashMap<String, Option<String>>,
}

impl Options {
    /// Parses the command line. `extra` lists the options specific to this
    /// benchmark. Also sets up the global thread pool.
    pub fn parse(
        name: &'static str,
        default_log_n: &[usize],
        default_repeat: usize,
        extra: &[(&str, Arg)],
    ) -> Self {
        let mut opts = Self {
            name,
            log_n: default_log_n.to_vec(),
            repeat: default_repeat,
            threads: 0,
            out_dir: PathBuf::from("results"),
            extra: HashMap::new(),
        };
        let mut threads = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
            match arg.as_str() {
                "--log-n" => opts.log_n = parse_list(&value()),
                "--repeat" => opts.repeat = value().parse().expect("--repeat takes an integer"),
                "--threads" => threads = Some(value().parse().expect("--threads takes an integer")),
                "--out-dir" => opts.out_dir = PathBuf::from(value()),
                // `cargo bench` passes this to every benchmark.
                "--bench" => {},
                other => match extra.iter().find(|(flag, _)| *flag == other) {
                    Some((_, Arg::Value)) => {
                        let v = value();
                        opts.extra.insert(other.to_string(), Some(v));
                    },
                    Some((_, Arg::Switch)) => {
                        opts.extra.insert(other.to_string(), None);
                    },
                    None => panic!("unknown option {other}"),
                },
            }
        }
        if let Some(threads) = threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .expect("the thread pool was already set up");
        }
        opts.threads = rayon::current_num_threads();
        opts
    }

    /// The value of a list option such as `--log-d 0,2,4`, or `default`.
    pub fn list(&self, flag: &str, default: &[usize]) -> Vec<usize> {
        match self.extra.get(flag) {
            Some(Some(v)) => parse_list(v),
            _ => default.to_vec(),
        }
    }

    /// The value of an integer option, or `default`.
    pub fn usize(&self, flag: &str, default: usize) -> usize {
        match self.extra.get(flag) {
            Some(Some(v)) => v.parse().unwrap_or_else(|_| panic!("{flag} takes an integer")),
            _ => default,
        }
    }

    /// The value of a text option, or `default`.
    pub fn text(&self, flag: &str, default: &str) -> String {
        match self.extra.get(flag) {
            Some(Some(v)) => v.clone(),
            _ => default.to_string(),
        }
    }

    /// Whether a switch was given.
    pub fn switch(&self, flag: &str) -> bool {
        self.extra.contains_key(flag)
    }
}

/// Parses `1,2,3` or a range `10..16` (inclusive), optionally with a step:
/// `10..22:2`.
fn parse_list(s: &str) -> Vec<usize> {
    s.split(',')
        .flat_map(|part| {
            let part = part.trim();
            if let Some((range, step)) = part.split_once("..") {
                let (end, step) = match step.split_once(':') {
                    Some((end, step)) => (end, step.parse().expect("bad step")),
                    None => (step, 1),
                };
                let (start, end): (usize, usize) =
                    (range.parse().expect("bad range"), end.parse().expect("bad range"));
                (start..=end).step_by(step).collect::<Vec<_>>()
            } else {
                vec![part.parse().expect("expected an integer")]
            }
        })
        .collect()
}

/// Runs `f` single-threaded.
pub fn single_threaded<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(f)
}

/// Times `f`.
pub fn time<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed())
}

/// The measured runs of one benchmark, written out when it is dropped.
pub struct Recorder {
    name: &'static str,
    /// The parameter columns, e.g. `scheme`, `log_n`, `d`, `threads`, `phase`.
    params: Vec<&'static str>,
    rows: Vec<(Vec<String>, usize, Duration)>,
    /// Groups (distinct parameter values) in first-seen order.
    groups: Vec<Vec<String>>,
    printed: usize,
    stem: PathBuf,
    started: String,
    command: String,
    threads: usize,
}

impl Recorder {
    /// Starts recording, with the given parameter columns.
    pub fn new(opts: &Options, params: &[&'static str]) -> Self {
        let started = utc_now();
        let stem = opts
            .out_dir
            .join(format!("{}-{}", opts.name, started.replace(':', "-")));
        let args: Vec<String> = std::env::args().skip(1).filter(|a| a != "--bench").collect();
        let command = format!("cargo bench --bench {} -- {}", opts.name, args.join(" "));
        println!(
            "{}: {} threads; writing {}.csv",
            opts.name,
            opts.threads,
            stem.display()
        );
        println!("{}", summary_header(params));
        Self {
            name: opts.name,
            params: params.to_vec(),
            rows: vec![],
            groups: vec![],
            printed: 0,
            stem,
            started,
            command,
            threads: opts.threads,
        }
    }

    /// Records one run. `params` must match the parameter columns.
    pub fn record(&mut self, params: Vec<String>, run: usize, time: Duration) {
        assert_eq!(params.len(), self.params.len());
        if !self.groups.contains(&params) {
            self.groups.push(params.clone());
        }
        self.rows.push((params, run, time));
    }

    /// Prints the summary of every group recorded since the last call.
    pub fn print_progress(&mut self) {
        for group in &self.groups[self.printed..] {
            println!("{}", summary_line(group, &self.times(group)));
        }
        self.printed = self.groups.len();
    }

    fn times(&self, group: &[String]) -> Vec<Duration> {
        self.rows
            .iter()
            .filter(|(p, ..)| p.as_slice() == group)
            .map(|(.., t)| *t)
            .collect()
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.print_progress();
        if let Some(dir) = self.stem.parent() {
            let _ = fs::create_dir_all(dir);
        }

        let mut csv = format!("{},run,seconds\n", self.params.join(","));
        for (params, run, time) in &self.rows {
            let _ = writeln!(csv, "{},{run},{:.9}", params.join(","), time.as_secs_f64());
        }

        let mut md = format!("# {}, {}\n\n", self.name, self.started.replace('T', " "));
        let _ = writeln!(md, "- Command: `{}`", self.command);
        let _ = writeln!(md, "- Commit: {}", git_commit());
        let _ = writeln!(md, "- Machine: {}", machine());
        let _ = writeln!(md, "- Threads: {}", self.threads);
        let _ = writeln!(md, "- Data: `{}`\n", file_name(&self.stem, "csv"));
        let mut header = self.params.clone();
        header.extend(["runs", "median", "mean"]);
        let _ = writeln!(md, "| {} |", header.join(" | "));
        let _ = writeln!(md, "|{}", "---|".repeat(header.len()));
        for group in &self.groups {
            let times = self.times(group);
            let (median, mean) = stats(&times);
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} |",
                group.join(" | "),
                times.len(),
                human(median),
                human(mean)
            );
        }

        for (ext, text) in [("csv", csv), ("md", md)] {
            let path = self.stem.with_extension(ext);
            match fs::write(&path, text) {
                Ok(()) => println!("wrote {}", path.display()),
                Err(e) => eprintln!("could not write {}: {e}", path.display()),
            }
        }
    }
}

fn summary_header(params: &[&str]) -> String {
    let mut cols: Vec<String> = params.iter().map(|p| format!("{p:<14}")).collect();
    cols.extend(["runs  ".into(), "median     ".into(), "mean".into()]);
    cols.join(" ")
}

fn summary_line(group: &[String], times: &[Duration]) -> String {
    let (median, mean) = stats(times);
    let mut cols: Vec<String> = group.iter().map(|p| format!("{p:<14}")).collect();
    cols.push(format!("{:<6}", times.len()));
    cols.push(format!("{:<11}", human(median)));
    cols.push(human(mean));
    cols.join(" ")
}

/// The median and mean.
fn stats(times: &[Duration]) -> (Duration, Duration) {
    let mut sorted = times.to_vec();
    sorted.sort();
    let mean = sorted.iter().sum::<Duration>() / sorted.len().max(1) as u32;
    (sorted[sorted.len() / 2], mean)
}

/// A duration with a readable unit.
pub fn human(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s >= 1.0 {
        format!("{s:.2} s")
    } else if s >= 1e-3 {
        format!("{:.2} ms", s * 1e3)
    } else {
        format!("{:.1} us", s * 1e6)
    }
}

fn file_name(stem: &std::path::Path, ext: &str) -> String {
    stem.with_extension(ext)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn git_commit() -> String {
    match command_output("git", &["rev-parse", "--short", "HEAD"]) {
        Some(commit) => {
            let dirty = command_output("git", &["status", "--porcelain", "--untracked-files=no"])
                .map_or(false, |s| !s.is_empty());
            if dirty {
                format!("{commit} (with uncommitted changes)")
            } else {
                commit
            }
        },
        None => "unknown".into(),
    }
}

fn machine() -> String {
    let cpu = command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(|| {
            fs::read_to_string("/proc/cpuinfo").ok().and_then(|info| {
                info.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split_once(':'))
                    .map(|(_, v)| v.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unknown CPU".into());
    let cores = std::thread::available_parallelism().map_or(0, |n| n.get());
    format!("{cpu}, {cores} cores")
}

/// The current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
fn utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}
