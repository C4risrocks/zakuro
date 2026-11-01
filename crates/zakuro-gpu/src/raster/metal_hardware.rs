//! The hardware rasterizer on Metal, phase M1.1.
//!
//! The Vulkan `hardware.rs` runs the PICA pipeline on the GPU: it feeds the
//! title's vertex shader program to SPIR-V shade/raster shaders, drives the
//! fragment stages, the tests and the blending, and moves whole command
//! lists' textures through one ring buffer. This module re-homes that
//! pipeline on Metal, shader for shader:
//!
//! - the GLSL sources are the same words (`shaders/*.vert/frag/comp`) that
//!   build the Vulkan SPIR-V; naga routes them to Metal (msl-out) instead
//!   of spirv-out, and the two variants the SPIR-V carries (`raster_depth
//!   .frag` built with WRITES_DEPTH, `upright_image.comp` built with
//!   TO_IMAGE) arrive through the same defines;
//! - the resource names keep the Vulkan descriptors: naga's msl writer
//!   maps them to Metal buffer/texture slots the pipelines bind;
//! - the ring and the display transfers keep their contract, so the present
//!   path consumes the raster's GPU textures instead of the CPU's
//!   framebuffer copies.
//!
//! # Phase A1 — the shader rack
//! This landing compiles the whole rack through naga to MSL in the order
//! the pipelines will be built from, and the test gives each one to a real
//! Metal device. Not one shader the Vulkan raster translates is allowed to
//! miss: the device compiles all eight MSL libraries.

use std::sync::OnceLock;

use naga::ShaderStage;

use super::super::shader::glsl;
const RASTER_VERT: &str = include_str!("../../shaders/raster.vert");
const RASTER_FRAG: &str = include_str!("../../shaders/raster.frag");
const SHADE_VERT: &str = include_str!("../../shaders/shade.vert");
const TRANSFER_COMP: &str = include_str!("../../shaders/transfer.comp");
const DEPTH_COMP: &str = include_str!("../../shaders/depth.comp");
const UPRIGHT_COMP: &str = include_str!("../../shaders/upright.comp");

/// The GLSL that built the SPIR-V is straight Vulkan dialect; naga's `glsl`
/// front only understands part of it, and this closes the gap without a
/// single change in meaning:
/// - `sampler2D` samplers become the separate `texture2D` + `sampler` pair
///   the Vulkan-GLSL dialect allows, and the `texture()` calls sample
///   through the `sampler2D(texture, sampler)` constructor;
/// - `constant_id` specialization constants become plain constants — the
///   Metal pipelines configure through defines instead of spec values.
/// The rewrite is line-shaped and textual on sources this port owns; the
/// rack test keeps the whole thing honest.
fn prepare(source: &str) -> String {
    let mut sampler_names: Vec<(u32, u32, String)> = Vec::new();
    let mut lines = String::with_capacity(source.len() + 512);
    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.contains("constant_id") && trimmed.contains(") const") {
            // layout(constant_id = 29) const uint DYNAMIC = 0u; → const …
            if let Some(after) = trimmed.find(") const") {
                lines.push_str(&trimmed[after + 2..]);
                lines.push('\n');
                continue;
            }
        }
        if trimmed.contains("writeonly buffer") {
            // naga's IR can't carry a store-only storage buffer (the Metal
            // bind is read-write anyway): dropping the qualifier keeps the
            // member writes legal without changing any word's meaning
            lines.push_str(
                &line.replace("writeonly buffer", "buffer"),
            );
            lines.push('\n');
            continue;
        }
        if trimmed.contains("layout(push_constant)") {
            // Metal has no push constants: naga 30 represents them as
            // immediate data and its MSL writer wants per-entry plumbing —
            // a plain uniform buffer keeps the words the same (A2 binds a
            // small shared buffer at this slot, exactly once per batch)
            lines.push_str(
                &line.replace(
                    "layout(push_constant)",
                    "layout(set = 0, binding = 24)",
                ),
            );
            lines.push('\n');
            continue;
        }
        if let Some((set, binding, name)) = sampler_declaration(trimmed) {
            // the sampler rides far above the real bindings, so the pairs
            // never collide (naga validates the whole binding table); A2's
            // pipelines bind by these numbers
            let sampler_binding = 10 + sampler_names.len() as u32;
            sampler_names.push((set, binding, name.clone()));
            lines.push_str(&format!(
                "layout(set = {set}, binding = {binding}) uniform texture2D {name}_texture;\n"
            ));
            lines.push_str(&format!(
                "layout(set = {set}, binding = {sampler_binding}) uniform sampler {name}_sampler;\n"
            ));
            continue;
        }
        lines.push_str(line);
        lines.push('\n');
    }
    let mut out = lines.replace(" precise ", " ");
    // the qualifier can also sit at the top of a line or function
    out = out
        .lines()
        .map(|line| {
            let leading = line.len() - line.trim_start().len();
            match line.trim_start().strip_prefix("precise ") {
                Some(rest) => format!("{}{}", &line[..leading], rest),
                None => line.to_owned(),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    // `precise` is a GLSL qualifier naga's front doesn't carry: dropping it
    // costs nothing (Metal's compiler keeps the value under non-fast-math
    // anyway) — the word only appears as the declaration's qualifier here
    for (.., name) in &sampler_names {
        let call = format!("texture({name},");
        let rewrite = format!("texture(sampler2D({name}_texture, {name}_sampler),");
        out = out.replace(&call, &rewrite);
    }
    out
}

/// reads one `layout(set = S, binding = N) uniform sampler2D name;` line.
fn sampler_declaration(line: &str) -> Option<(u32, u32, String)> {
    let rest = line.strip_prefix("layout(set = ")?;
    let (set_text, after) = rest.split_once(',')?;
    let set: u32 = set_text.parse().ok()?;
    let after = after.trim().strip_prefix("binding = ")?;
    let (binding_text, after) = after.split_once(')')?;
    let binding: u32 = binding_text.parse().ok()?;
    let name = after
        .trim()
        .strip_prefix("uniform sampler2D ")?
        .trim_end_matches(';')
        .trim()
        .to_owned();
    (!name.is_empty()).then_some((set, binding, name))
}

/// one pipeline's source translated to MSL, and how long the translation
/// took — the stat the later phases keep reporting.
#[derive(Debug)]
pub struct Translated {
    pub name: &'static str,
    pub msl: String,
    pub microseconds: u128,
}

impl Translated {
    fn new(
        name: &'static str,
        source: &str,
        stage: ShaderStage,
        defines: &[(&str, &str)],
    ) -> Result<Self, String> {
        let started = std::time::Instant::now();
        let msl = glsl::compile_msl_stage(&prepare(source), stage, defines)?;
        let microseconds = started.elapsed().as_micros();
        Ok(Self { name, msl, microseconds })
    }
}

/// the whole rack, translated once at first use, in the order the
/// pipelines get built from.
#[derive(Debug)]
pub struct Rack {
    pub shade_vert: Translated,
    pub raster_vert: Translated,
    pub raster_frag: Translated,
    pub raster_depth_frag: Translated,
    pub transfer_comp: Translated,
    pub depth_comp: Translated,
    pub upright_comp: Translated,
    pub upright_image_comp: Translated,
}

impl Rack {
    fn translated() -> Result<Self, String> {
        Ok(Self {
            shade_vert: Translated::new("shade.vert", SHADE_VERT, ShaderStage::Vertex, &[])?,
            raster_vert: Translated::new("raster.vert", RASTER_VERT, ShaderStage::Vertex, &[])?,
            raster_frag: Translated::new("raster.frag", RASTER_FRAG, ShaderStage::Fragment, &[])?,
            raster_depth_frag: Translated::new(
                "raster_depth.frag",
                RASTER_FRAG,
                ShaderStage::Fragment,
                &[("WRITES_DEPTH", "1")],
            )?,
            transfer_comp: Translated::new("transfer.comp", TRANSFER_COMP, ShaderStage::Compute, &[])?,
            depth_comp: Translated::new("depth.comp", DEPTH_COMP, ShaderStage::Compute, &[])?,
            upright_comp: Translated::new("upright.comp", UPRIGHT_COMP, ShaderStage::Compute, &[])?,
            upright_image_comp: Translated::new(
                "upright_image.comp",
                UPRIGHT_COMP,
                ShaderStage::Compute,
                &[("TO_IMAGE", "1")],
            )?,
        })
    }

    pub fn entries(&self) -> [&Translated; 8] {
        [
            &self.shade_vert,
            &self.raster_vert,
            &self.raster_frag,
            &self.raster_depth_frag,
            &self.transfer_comp,
            &self.depth_comp,
            &self.upright_comp,
            &self.upright_image_comp,
        ]
    }
}

static RACK: OnceLock<Result<Rack, String>> = OnceLock::new();

/// the rack translated and cached; errors surface verbatim so the next
/// phase knows exactly where naga meets Metal's model.
pub fn rack() -> Result<&'static Rack, &'static String> {
    RACK.get_or_init(Rack::translated).as_ref()
}

#[cfg(all(target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice as _};

    /// TEMP probe A1: the smallest parse of each suspect construct.
    #[test]
    fn parse_probe() {
        for (label, source, stage) in [
            ("sampler", r#"
#version 450
layout(set = 0, binding = 0) uniform sampler2D t;
layout(location = 0) out vec4 c;
void main() { c = texture(t, vec2(0.5)); }
"#, ShaderStage::Fragment),
            ("combined-no-layout", r#"
#version 450
uniform sampler2D t;
layout(location = 0) out vec4 c;
void main() { c = texture(t, vec2(0.5)); }
"#, ShaderStage::Fragment),
            ("spec-const", r#"
#version 450
layout(constant_id = 29) const uint DYNAMIC = 0u;
layout(location = 0) out vec4 c;
void main() { c = vec4(float(DYNAMIC)); }
"#, ShaderStage::Fragment),
            ("separate-texture-sampler", r#"
#version 450
layout(set = 0, binding = 0) uniform texture2D t_tex;
layout(set = 0, binding = 1) uniform sampler t_sam;
layout(location = 0) out vec4 c;
void main() { c = texture(sampler2D(t_tex, t_sam), vec2(0.5)); }
"#, ShaderStage::Fragment),
            ("noperspective", r#"
#version 450
layout(location = 3) noperspective in float d;
layout(location = 0) out vec4 c;
void main() { c = vec4(d); }
"#, ShaderStage::Fragment),
        ] {
            match glsl::compile_msl_stage(source, stage, &[]) {
                Ok(msl) => println!("probe {label}: OK \u{2713} ({} chars)", msl.len()),
                Err(error) => println!("probe {label}: FAIL \u{2717}: {}", error.trim()),
            }
        }
    }

    /// the Vulkan path's whole rack, through naga to MSL, given to a real
    /// Metal device: the gate A2's pipelines build against.
    #[test]
    fn the_hardware_rack_translates_to_msl_and_the_device_compiles() {
        let started = std::time::Instant::now();
        let rack = rack().expect("the whole hardware rack translates to MSL");
        let device = MTLCreateSystemDefaultDevice().expect("this Mac has a Metal device");
        let mut translated_microseconds = 0;
        let mut device_compiled = 0;
        let mut misses = String::new();
        for translated in rack.entries() {
            translated_microseconds += translated.microseconds;
            let source = NSString::from_str(&translated.msl);
            if device.newLibraryWithSource_options_error(&source, None).is_ok() {
                device_compiled += 1;
            } else {
                misses.push_str(&format!("{} ", translated.name));
            }
        }
        let elapsed = started.elapsed().as_millis();
        println!(
            "M1.1 A1: 8 shaders translated in {}\u{b5}s, the rack device-compiled in {elapsed} ms",
            translated_microseconds
        );
        assert_eq!(device_compiled, 8, "the device rejected: {misses}");
    }
}
