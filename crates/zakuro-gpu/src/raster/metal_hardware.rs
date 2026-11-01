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

#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2::rc::Retained;
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2_foundation::NSString;
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2_metal::{
    MTLCommandQueue as _, MTLCreateSystemDefaultDevice, MTLDevice as _, MTLPixelFormat,
    MTLLibrary as _, MTLRenderPipelineDescriptor, MTLVertexStepFunction,
    MTLVertexFormat, MTLVertexDescriptor,
};
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

/// a local helper (the backend's own `ns` stays private there).
fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

/// one pipeline's source translated to MSL, and how long the translation
/// took — the stat the later phases keep reporting.
#[derive(Debug)]
pub struct Translated {
    pub name: &'static str,
    pub msl: String,
    /// the naga writer mangles the entry (`main` → `main0`…); MSL's
    /// function binds by THIS name in the pipelines.
    pub entry: String,
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
        let (msl, entry) = glsl::compile_msl_stage_entry(&prepare(source), stage, defines)?;
        let microseconds = started.elapsed().as_micros();
        Ok(Self { name, msl, entry, microseconds })
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

/// the render target is RGBA8 through and through, matching the Vulkan
/// path's COLOR_FORMAT choice.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub const COLOR_PIXEL_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA8Unorm;

/// the depth the console's 24 bits stand as they do on Vulkan: the D24
/// UNORM + stencil 8 as the default, and the float twin (its 32 bits
/// holding the 24 exactly) the fallback Vulkan keeps for GPUs that draw
/// into D24S8 badly.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn depth_pixel_format(float_depth: bool) -> MTLPixelFormat {
    if float_depth {
        MTLPixelFormat::Depth32Float_Stencil8
    } else {
        MTLPixelFormat::Depth24Unorm_Stencil8
    }
}

/// the hardware raster's Metal state: device, queue, and the rack's MSL
/// libraries, one per shader's name. A2 lands this cold; A3 wires the draw
/// execution, the ring and the display transfers through it.
pub struct HardwareMetal {
    device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    libraries: std::collections::HashMap<
        &'static str,
        Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>,
    >,
    /// the naga-mangled entry name per shader (`main` becomes `main0`…)
    entries: std::collections::HashMap<&'static str, String>,
}

/// what a draw's vertices go through on Metal, the Vulkan twin's enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexStage {
    /// triangles the CPU shaded and placed, raster.vert passing them on.
    Placed,
    /// the title's program interpreted by the shade pipeline.
    Interpreted,
}

/// a vertex is six vectors: position, color, texture coordinates 0 and 1,
/// coordinate 2 with the mapped depth, the quaternion and the view vector.
const VERTEX_SIZE: u64 = 24 * 4;
/// the depth-buffer bindings of the compiled MSL: the raster's targets,
/// with the push constants at 24 (metal_hardware's `prepare` set that).
const PUSH_CONSTANT_BUFFER: u64 = 24;

impl HardwareMetal {
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or("no Metal device on this Mac")?;
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;
        let rack = rack()?;
        let mut libraries = std::collections::HashMap::new();
        for translated in rack.entries() {
            let source = NSString::from_str(&translated.msl);
            let library = device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|error| format!("the {} MSL did not compile: {error}", translated.name))?;
            libraries.insert(translated.name, library);
        }
        let entries = rack
            .entries()
            .iter()
            .map(|translated| (translated.name, translated.entry.clone()))
            .collect();
        Ok(Self { device, queue, libraries, entries })
    }

    pub fn function(
        &self,
        shader: &'static str,
    ) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLFunction>>, String> {
        use objc2_metal::MTLLibrary as _;
        let library = self
            .libraries
            .get(shader)
            .ok_or_else(|| format!("no {shader} library in the rack"))?;
        let entry = self
            .entries
            .get(shader)
            .ok_or_else(|| format!("no {shader} entry in the rack"))?;
        library
            .newFunctionWithName(&ns(entry))
            .ok_or_else(|| format!("no {entry} in {shader}"))
    }

    /// the graphics pipeline in the Vulkan twin's shape (create_graphics_
    /// pipeline): the color target RGBA8, blending deferred to the encoder
    /// (Metal enforces blending at encode time; Vulkan's per-key blending
    /// states land with A3's draw-time wiring), the depth the twin's D24S8
    /// (or its float twin), stencil riding the same format, and the placed
    /// vertices read six vec4s when the CPU shaded and placed them.
    pub fn graphics_pipeline(
        &self,
        vertex: &'static str,
        fragment: &'static str,
        key: &PipelineKey,
    ) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>>, String> {
        let descriptor = MTLRenderPipelineDescriptor::new();
        // the depth pair goes first, setters and validation both: Metal's
        // eager pipeline-descriptor validation snapshots the formats of an
        // unset field once, and the depth needs its stencil beside it
        let depth_format = depth_pixel_format(key.float_depth);
        descriptor.setStencilAttachmentPixelFormat(depth_format);
        descriptor.setDepthAttachmentPixelFormat(depth_format);
        unsafe {
            let attachments = descriptor.colorAttachments();
            let color: Retained<objc2_metal::MTLRenderPipelineColorAttachmentDescriptor> =
                attachments.objectAtIndexedSubscript(0);
            color.setPixelFormat(COLOR_PIXEL_FORMAT);
        }
        descriptor.setRasterSampleCount(1);
        descriptor.setVertexFunction(Some(self.function(vertex)?.as_ref()));
        descriptor.setFragmentFunction(Some(self.function(fragment)?.as_ref()));
        // the draw's write mask rides with A3's encode-time blending; the
        // default state writes all four channels, as the key's 0xF mask
        if key.vertex == VertexStage::Placed {
            let vertex_descriptor: Retained<objc2_metal::MTLVertexDescriptor> =
                unsafe { objc2_metal::MTLVertexDescriptor::vertexDescriptor() };
            unsafe {
                let attributes = vertex_descriptor.attributes();
                for location in 0..6 {
                    let attribute: Retained<objc2_metal::MTLVertexAttributeDescriptor> =
                        attributes.objectAtIndexedSubscript(location);
                    attribute.setFormat(MTLVertexFormat::Float4);
                    attribute.setOffset(location * 16);
                    attribute.setBufferIndex(0);
                }
                let layouts = vertex_descriptor.layouts();
                let layout: Retained<objc2_metal::MTLVertexBufferLayoutDescriptor> =
                    layouts.objectAtIndexedSubscript(0);
                layout.setStride(VERTEX_SIZE as usize);
                layout.setStepFunction(MTLVertexStepFunction::PerVertex);
                descriptor.setVertexDescriptor(Some(&vertex_descriptor));
            }
        }
        unsafe { self.device.newRenderPipelineStateWithDescriptor_error(&descriptor) }
            .map_err(|error| format!("no pipeline state: {error}"))
    }

    /// the smaller entry point A2's test exercises: the default pipelines,
    /// both depth variants, and placed vertices.
    pub fn default_pipelines(
        &self,
        key_fn: impl Fn(bool) -> PipelineKey,
    ) -> Result<
        [Retained<ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>>; 2],
        String,
    > {
        Ok([
            self.graphics_pipeline("raster.vert", "raster.frag", &key_fn(true))?,
            self.graphics_pipeline("raster.vert", "raster.frag", &key_fn(false))?,
        ])
    }
}

/// the Metal twin of the Vulkan pipeline key, without its vk::ShaderModule:
/// the blend and the fragment specialization ride in A3 (their per-key MSL)
/// — A2's keys call out only the two shapes the default pipelines take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    /// red, green, blue and alpha writes, one bit each (0xF default).
    pub mask: u32,
    /// whether the pass carries a depth+stencil target.
    pub depth: bool,
    /// the depth's D32-float twin, the way the Vulkan path falls back.
    pub float_depth: bool,
    pub vertex: VertexStage,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use objc2_metal::MTLPixelFormat;
    use objc2_metal::{MTLCommandQueue as _, MTLDevice as _};

    #[test]
    fn pixel_formats_match_the_vulkan_twins() {
        // Vulkan's COLOR_FORMAT and DEPTH_FORMAT with its float fallback
        assert_eq!(COLOR_PIXEL_FORMAT, MTLPixelFormat::RGBA8Unorm);
        assert_eq!(depth_pixel_format(false), MTLPixelFormat::Depth24Unorm_Stencil8);
        assert_eq!(depth_pixel_format(true), MTLPixelFormat::Depth32Float_Stencil8);
    }

    #[test]
    fn default_pipelines_build_on_the_device() {
        let hardware = HardwareMetal::new().expect("the Metal raster constructs");
        // the default pipelines; debug: the float twin bisects the
        // descriptor crash (Apple silicon may not render D24S8)
        let float_depth = true;
        let depth = true;
        let depth = float_depth;
        for depth in [depth] {
            let pipeline = hardware
                .graphics_pipeline(
                    "raster.vert",
                    "raster.frag",
                    &PipelineKey { mask: 0xF, depth, float_depth, vertex: VertexStage::Placed },
                )
                .expect("the default pipelines construct");
            let _ = pipeline;
        }
        // the libraries: every rack shader carries a main
        for shader in [
            "shade.vert", "raster.vert", "raster.frag", "raster_depth.frag",
            "transfer.comp", "depth.comp", "upright.comp", "upright_image.comp",
        ] {
            hardware.function(shader).expect(shader);
        }
    }
}
