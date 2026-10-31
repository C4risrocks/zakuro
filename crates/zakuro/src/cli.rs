//! command line parsing.

use zakuro_gpu::RendererKind;
use crate::settings::Language;

#[derive(Debug, Clone)]
pub struct Options {
    /// the game to start, none to open the library.
    pub rom: Option<String>,
    /// the settings' choice when not given.
    pub renderer: Option<RendererKind>,
    /// window scale relative to the console's 400x480 combined screens, the
    /// settings' when not given.
    pub scale: Option<u32>,
    /// runtime override; never written into player settings.
    pub language: Option<Language>,
    /// run without a window for this many frames, then report.
    pub headless: Option<u64>,
    pub profile: bool,
    pub new3ds: bool,
    /// skip loading a ROM entirely and just paint both screens solid colors.
    pub test_pattern: bool,
    pub test_frames: Option<u64>,
    /// the split-monitors spike: a second window draws the bottom screen
    /// alone, straight from the same frames.
    pub split_spike: bool,
    /// a second window draws one screen alone, on another monitor when
    /// there is one, this run only.
    pub split: Option<bool>,
    /// which monitor the second window goes to, the desktop's order by
    /// number.
    pub monitor_bottom: Option<usize>,
    /// a library of recompiled code for the title, or a directory of them.
    pub recompiled: Option<String>,
    /// draw on the host GPU rather than in software, the settings' choice
    /// when not given.
    pub hardware_rasterizer: Option<bool>,
    /// where saves live, instead of the usual place.
    pub data: Option<String>,
    /// interpret everything, whatever recompiled code there is.
    pub interpreter: bool,
    /// play no sound.
    pub mute: bool,
}

const USAGE: &str = "\
zakuro - a high-level-emulation Nintendo 3DS emulator

usage: zakuro [rom.3ds|.cxi] [options]

Without a game it opens the library, the games in a folder you pick. The
options below override the settings for this run.

options:
  --renderer <vulkan|gl|software>  presentation backend (default: vulkan,
                                   or gl where Vulkan does not start)
  --rasterizer <hardware|software> where the 3D is drawn (default: hardware,
                                   the host GPU through Vulkan, or software
                                   where that does not start)
  --language <code>                system language for this run (ja,en,fr,de,it,es,
                                   zh-Hans,ko,nl,pt,ru,zh-Hant)
  --scale <n>                      window scale factor (default: 2)
  --split <on|off>                 a second window draws one screen alone
                                   (the bottom one), this run only
  --monitor-bottom <n>             which monitor it goes to, the desktop's
                                   order by number, 0 first
  --split-spike                    --split on, the spike's older name
  (Esc opens the menu over a game, F1 pauses, F11 goes fullscreen)
  --test-pattern                   synthetic red/blue screens, no game data
  --test-frames <n>                exit after n presentations (test-pattern only)
  --headless <frames>              run without a window and print a report
  --new3ds                         emulate a New 3DS
  --profile                        collect a sampling profile and print it
  --recompiled <path>              run code 3dsrecomp built for the title, a
                                   library or a directory holding <title id>.so
                                   (.dll on Windows, default: the one 3dsrecomp
                                   build installed for it, in the system's
                                   place for data, ~/.local/share/3dsrecomp or
                                   %APPDATA%\\3dsrecomp)
  --interpreter                    interpret everything, even with recompiled
                                   code around
  --mute                           play no sound
  --data <dir>                     where saves live (default: the system's
                                   place for data, ~/.local/share/zakuro or
                                   %APPDATA%\\zakuro)
  -h, --help                       show this message

The system font cannot be generated: put a dump at sysdata/shared_font.bin, in
the working directory or where saves live, for titles that render text with it.
";

pub fn parse() -> Result<Options, String> {
    parse_args(std::env::args().skip(1))
}

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut rom = None;
    let mut options = Options {
        rom: None,
        renderer: None,
        scale: None,
        language: None,
        headless: None,
        profile: false,
        new3ds: false,
        test_pattern: false,
        test_frames: None,
        split_spike: false,
        split: None,
        monitor_bottom: None,
        recompiled: None,
        hardware_rasterizer: None,
        data: None,
        interpreter: false,
        mute: false,
    };

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--renderer" => {
                let value = args.next().ok_or("--renderer needs a value")?;
                options.renderer = Some(
                    RendererKind::parse(&value).ok_or_else(|| format!("unknown renderer '{value}'"))?,
                );
            }
            "--language" => {
                let value = args.next().ok_or("--language needs a code")?;
                options.language = Some(Language::parse(&value).ok_or_else(|| format!("unknown language '{value}'"))?);
            }
            "--scale" => {
                let value = args.next().ok_or("--scale needs a value")?;
                options.scale = Some(value.parse().map_err(|_| format!("'{value}' is not a scale"))?);
            }
            "--headless" => {
                let value = args.next().ok_or("--headless needs a frame count")?;
                options.headless = Some(
                    value
                        .parse()
                        .map_err(|_| format!("'{value}' is not a frame count"))?,
                );
            }
            "--new3ds" => options.new3ds = true,
            "--rasterizer" => {
                let value = args.next().ok_or("--rasterizer needs a value")?;
                options.hardware_rasterizer = Some(match value.as_str() {
                    "hardware" | "vulkan" | "gpu" => true,
                    "software" | "cpu" => false,
                    other => return Err(format!("unknown rasterizer '{other}'")),
                });
            }
            "--recompiled" => {
                options.recompiled = Some(args.next().ok_or("--recompiled needs a path")?);
            }
            "--data" => {
                options.data = Some(args.next().ok_or("--data needs a directory")?);
            }
            "--interpreter" => options.interpreter = true,
            "--mute" => options.mute = true,
            "--profile" => options.profile = true,
            "--test-pattern" => options.test_pattern = true,
            "--test-frames" => {
                let value = args.next().ok_or("--test-frames needs a count")?;
                let frames = value.parse().map_err(|_| "invalid test frame count")?;
                if frames == 0 { return Err("--test-frames must be positive".into()); }
                options.test_frames = Some(frames);
            },
            "--split-spike" => options.split_spike = true,
            "--split" => {
                let value = args.next().ok_or("--split needs a value")?;
                options.split = Some(match value.as_str() {
                    "on" | "y" | "1" => true,
                    "off" | "n" | "0" => false,
                    other => return Err(format!("unknown split '{other}', use on or off")),
                });
            }
            "--monitor-bottom" => {
                let value = args.next().ok_or("--monitor-bottom needs a value")?;
                options.monitor_bottom =
                    Some(value.parse().map_err(|_| format!("'{value}' is not a monitor number"))?);
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option '{other}'"));
            }
            path => rom = Some(path.to_owned()),
        }
    }

    options.rom = rom;
    if options.test_frames.is_some() && !options.test_pattern {
        return Err("--test-frames requires --test-pattern".into());
    }
    if options.headless.is_some() && options.rom.is_none() && !options.test_pattern {
        print!("{USAGE}");
        return Err("--headless needs a game".to_owned());
    }
    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Result<Options, String> {
        parse_args(values.iter().map(|value| value.to_string()))
    }
    #[test]
    fn explicit_split_off_overrides_saved_companion() {
        assert!(!args(&["--split", "off"]).unwrap().companion_wanted(true));
        assert!(args(&[]).unwrap().companion_wanted(true));
        assert!(!args(&[]).unwrap().companion_wanted(false));
    }
    #[test]
    fn language_override_uses_console_codes_without_settings_writes() {
        assert_eq!(args(&["--language", "es"]).unwrap().language.unwrap().code(), 5);
        assert_eq!(args(&["--language", "zh-Hant"]).unwrap().language.unwrap().code(), 11);
        assert!(args(&["--language", "bad"]).is_err());
        assert!(args(&["--language"]).is_err());
        assert!(args(&[]).unwrap().language.is_none());
    }
}

impl Options {
    pub fn companion_wanted(&self, saved: bool) -> bool {
        self.split.unwrap_or(self.split_spike || saved)
    }
}
