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
use naga::back::msl as naga_msl;

use super::super::shader::glsl;
use crate::format::ColorFormat;

#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2::rc::Retained;
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2_foundation::NSString;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::ptr::NonNull;
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2_metal::MTLBuffer as _;
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2_metal::{
    MTLClearColor, MTLLoadAction, MTLCommandQueue as _, MTLCreateSystemDefaultDevice,
    MTLDevice as _, MTLPixelFormat, MTLLibrary as _, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLVertexStepFunction, MTLVertexFormat,
    MTLVertexDescriptor, MTLTextureUsage, MTLStoreAction, MTLTexture as _,
    MTLCommandBuffer as _, MTLRenderCommandEncoder as _,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2::runtime::AnyObject;
const RASTER_VERT: &str = include_str!("../../shaders/raster.vert");
const RASTER_FRAG: &str = include_str!("../../shaders/raster.frag");
const SHADE_VERT: &str = include_str!("../../shaders/shade.vert");
const TRANSFER_COMP: &str = include_str!("../../shaders/transfer.comp");
const DEPTH_COMP: &str = include_str!("../../shaders/depth.comp");
const UPRIGHT_COMP: &str = include_str!("../../shaders/upright.comp");

/// the fragment raster's spec constants, by `constant_id`, the words of
/// the Vulkan pipeline's specialization — the values a Metal compile bakes
/// in (naga flattens them to plain constants).
pub const FRAGMENT_CONSTANTS: usize = 30;

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
fn prepare(
    source: &str,
    values: Option<&[u32; FRAGMENT_CONSTANTS]>,
    to_image: bool,
) -> String {
    let mut sampler_names: Vec<(u32, u32, String)> = Vec::new();
    let mut lines = String::with_capacity(source.len() + 512);
    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.contains("constant_id") && trimmed.contains(") const") {
            // layout(constant_id = 29) const uint DYNAMIC = 0u; → the
            // const with the value the pipeline's key baked in (naga's MSL
            // writer refuses spec overrides), or the source's own default
            if let Some(after) = trimmed.find(") const") {
                let head = &trimmed[after + 2..];
                if let Some(values) = values {
                    // `const uint NAME = value;` — the id ranks the spec
                    // word the pipeline key carries; the name rebuild without
                    // its own const (the head has it already)
                    let id: u32 = trimmed
                        .split("constant_id = ")
                        .nth(1)
                        .and_then(|id_part| id_part.split(')').next())
                        .and_then(|id| id.parse().ok())
                        .unwrap_or(0);
                    let value = values.get(id as usize).copied().unwrap_or(0);
                    let name = head.split('=').next().unwrap_or("").trim();
                    let line = format!("{name} = {value}u;");
                    lines.push_str(&line);
                    lines.push('\n');
                    continue;
                }
                lines.push_str(head);
                lines.push('\n');
                continue;
            }
        }
        if to_image && trimmed.contains("buffer Upright") {
            // the image variant writes the image, so the SSBO moves to a
            // free binding (the two shared Vulkan binding 1 otherwise)
            lines.push_str(
                &line.replace("binding = 1)", "binding = 2)"),
            );
            lines.push('\n');
            continue;
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

/// a draw's placed vertices packed the hardware.rs `place`'s own way: the
/// 24-float rows the vertex shader reads, from the CPU-placed `Screen`s.
pub(crate) fn pack_placed(
    vertices: &[super::Screen],
    width: u32,
    height: u32,
    depth_map: &super::DepthMap,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(vertices.len() * 24);
    for v in vertices {
        let w = 1.0 / v.inv_w;
        let x = v.x / width as f32 * 2.0 - 1.0;
        let y = v.y / height as f32 * 2.0 - 1.0;
        let depth = v.z * depth_map.scale + depth_map.offset;
        let (c, t, q, view) = (v.color_over_w, v.texcoords_over_w, v.quaternion_over_w, v.view_over_w);
        out.extend([
            x * w, y * w, 0.0, w,
            c[0] * w, c[1] * w, c[2] * w, c[3] * w,
            t[0][0] * w, t[0][1] * w, t[1][0] * w, t[1][1] * w,
            t[2][0] * w, t[2][1] * w, depth, 0.0,
            q[0] * w, q[1] * w, q[2] * w, q[3] * w,
            view[0] * w, view[1] * w, view[2] * w,
            v.texcoord0_w_over_w * w,
        ]);
    }
    out
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
        Self::keyed(name, source, stage, defines, None, None)
    }

    fn keyed(
        name: &'static str,
        source: &str,
        stage: ShaderStage,
        defines: &[(&str, &str)],
        values: Option<&[u32; FRAGMENT_CONSTANTS]>,
        bindings: Option<&naga_msl::EntryPointResourceMap>,
    ) -> Result<Self, String> {
        let started = std::time::Instant::now();
        let to_image = defines.iter().any(|(name, _)| *name == "TO_IMAGE");
        let (msl, entry) = glsl::compile_msl_stage_entry(
            &prepare(source, values, to_image),
            stage,
            defines,
            bindings,
        )?;
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

fn texture_target(slot: u8) -> naga_msl::BindTarget {
    naga_msl::BindTarget { texture: Some(slot), ..Default::default() }
}

fn buffer_target(slot: u8) -> naga_msl::BindTarget {
    naga_msl::BindTarget { buffer: Some(slot), ..Default::default() }
}

impl Rack {
    /// The per-entry-point binding maps: the resources' Vulkan bindings
    /// become the same-numbered Metal slots (the textures, buffers and
    /// samplers carry their own namespaces, so the numbers never clash),
    /// the immediates (the prepared push constants) ride slot 24, and the
    /// runtime-sized arrays take their host-size word at 31.
    fn shader_bindings(
        label: &'static str,
        resources: Vec<(u32, u8, bool)>,
        sizes: Option<u8>,
    ) -> Result<naga_msl::EntryPointResourceMap, String> {
        let mut map = naga_msl::BindingMap::default();
        for (binding, slot, is_texture) in resources {
            let target = if is_texture { texture_target(slot) } else { buffer_target(slot) };
            map.insert(naga::ResourceBinding { group: 0, binding }, target);
        }
        if !map.contains_key(&naga::ResourceBinding { group: 0, binding: 24 }) {
            // the prepared push constants: naga's Immediate address space
            // resolves through the map's entry binding when carried
            map.insert(
                naga::ResourceBinding { group: 0, binding: 24 },
                buffer_target(24),
            );
        }
        let mut per_entry = naga_msl::EntryPointResourceMap::default();
        per_entry.insert(String::from("main"), naga_msl::EntryPointResources {
            resources: map,
            immediates_buffer: None,
            sizes_buffer: sizes,
        });
        println!("bindings [{label}]: {:?}", per_entry.first_key_value().map(|(key, value)| {
            (key, value.resources.keys().map(|resource| resource.binding).collect::<Vec<_>>(), value.sizes_buffer)
        }));
        Ok(per_entry)
    }

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
            transfer_comp: Translated::keyed(
                "transfer.comp",
                TRANSFER_COMP,
                ShaderStage::Compute,
                &[],
                None,
                Some(&Self::shader_bindings(
                    "transfer",
                    vec![(0, 0, true), (1, 1, true)],
                    Some(30),
                )?),
            )?,
            depth_comp: Translated::keyed(
                "depth.comp",
                DEPTH_COMP,
                ShaderStage::Compute,
                &[],
                None,
                Some(&Self::shader_bindings(
                    "depth",
                    vec![(0, 0, false), (1, 1, false), (2, 2, true)],
                    Some(30),
                )?),
            )?,
            upright_comp: Translated::keyed(
                "upright.comp",
                UPRIGHT_COMP,
                ShaderStage::Compute,
                &[],
                None,
                Some(&Self::shader_bindings(
                    "upright",
                    vec![(0, 0, true), (1, 1, false)],
                    Some(30),
                )?),
            )?,
            upright_image_comp: Translated::keyed(
                "upright_image.comp",
                UPRIGHT_COMP,
                ShaderStage::Compute,
                &[("TO_IMAGE", "1")],
                None,
                Some(&Self::shader_bindings(
                    "upright-image",
                    vec![(0, 0, true), (1, 1, true), (2, 2, false)],
                    Some(30),
                )?),
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

mod pica;
pub use pica::OwnedTextureBinding;

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
    /// the draw inputs' ring (A4's batch staging) — one frame at a time.
    pub(crate) ring: Ring,
    /// the draw targets backed by the guest's memory (A4).
    pub(crate) surfaces: Vec<MetalSurface>,
    pica_textures: std::collections::HashMap<usize, pica::Texture>,
    pica_samplers: std::collections::HashMap<super::pica_state::Filtering, Retained<ProtocolObject<dyn objc2_metal::MTLSamplerState>>>,
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
        // 8 MB for the draw inputs — the Vulkan twin's 64 MB scaled down
        // first, the batch's own defines grow it honestly when A4's real
        // draws land inside
        let ring = Ring::new(device.as_ref(), 8 << 20)?;
        Ok(Self { device, queue, libraries, entries, ring, surfaces: Vec::new(), pica_textures: Default::default(), pica_samplers: Default::default() })
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

/// A4's foundation: the batch ring the draw inputs ride in, the way the
/// Vulkan raster's `RING_SIZE`/`RING_FLUSH`/`IN_FLIGHT` carry them. one
/// shared MTL buffer backs the ring; a batch allocates from the head and
/// holds its span until the GPU has run its command buffer, then the span
/// returns to the free pool. the flush rule: when drafting a batch would
/// walk the head past the widest live-byte span, the host flushes first —
/// the same guarantee the Vulkan path executes through its fences.
pub struct Ring {
    buffer: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    capacity: u64,
    /// the write head, bytes from the ring's start.
    head: u64,
    /// the live spans, oldest first, each awaiting its command buffer's
    /// completion: `(start, bytes)`.
    live: std::collections::VecDeque<(u64, u64)>,
    flushed: u64,
}

impl Ring {
    /// a ring over a fresh Shared buffer of the capacity asked. Metal's
    /// maxBufferLength on Apple silicon outsizes the 64 MB the Vulkan
    /// twin asks; `capacity` stays the caller's decision.
    pub fn new(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        capacity: u64,
    ) -> Result<Self, String> {
        use objc2_metal::{MTLBuffer as _, MTLResourceOptions};
        let buffer = unsafe {
            device.newBufferWithLength_options(capacity as usize, MTLResourceOptions::StorageModeShared)
        }.ok_or("the ring's buffer did not allocate")?;
        Ok(Self { buffer, capacity, head: 0, live: std::collections::VecDeque::new(), flushed: 0 })
    }

    /// the bytes a batch can take without the flush rule firing.
    pub fn free(&self) -> u64 {
        let live: u64 = self.live.iter().map(|(_, bytes)| bytes).sum();
        self.capacity - live.min(self.capacity)
    }

    /// takes a span from the head, wrapping to the ring's start. the
    /// caller writes the shared bytes directly; the span retires when its
    /// command buffer completes (`retire`).
    pub fn take(&mut self, bytes: u64) -> Result<(u64, NonNull<u8>), String> {
        if bytes > self.free() {
            return Err(format!(
                "the batch's {} bytes need a flush; only {} live-free",
                bytes, self.free()
            ));
        }
        // wrap: the head's tail span must not cross the ring's end
        let tail = self.capacity - self.head;
        if bytes > tail {
            // the head jumps to the start; the tail's bytes return unused
            self.head = 0;
        }
        let start = self.head;
        self.head += bytes;
        self.live.push_back((start, bytes));
        use objc2_metal::MTLBuffer as _;
        let pointer = unsafe {
            self.buffer.contents().cast::<u8>().add(start as usize)
        };
        Ok((start, pointer))
    }

    /// a batch's command buffer completed: its span gives back.
    pub fn retire(&mut self, start: u64, bytes: u64) {
        while let Some((front_start, front_bytes)) = self.live.pop_front() {
            if front_start == start && front_bytes == bytes {
                return;
            }
        }
    }

    /// the ring's buffer, for the callers writing bulk spans directly.
    pub fn buffer_of(&self) -> &ProtocolObject<dyn objc2_metal::MTLBuffer> {
        self.buffer.as_ref()
    }

    /// whether a flush should run before the next batch (the Vulkan
    /// twin's RING_FLUSH: half the ring's span walked).
    pub fn should_flush(&self) -> bool {
        self.free() < self.capacity / 2
    }
}

/// A4's texture staging: the decoded texels the shared `TextureCache`
/// hands over (the SAME decode layer the software path runs — the port
/// costs nothing there), keyed by the texels' own identity. A texture the
/// draw samples stays the Metal texture its texels were uploaded into
/// until a decode hands different ones — the Vulkan raster's
/// `textures: HashMap<usize, Texture>` by the texels' address, the same
/// trick through objc2's Managed storage.
#[derive(Default)]
pub struct TexturePool {
    textures: std::collections::HashMap<usize, PoolTexture>,
}

struct PoolTexture {
    texture: Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>,
    width: u32,
    height: u32,
}

impl TexturePool {
    pub fn new() -> Self {
        Self::default()
    }

    /// the texture the texels sample from, uploaded once and kept until
    /// the texels' identity changes. `key` is the texels' own address (the
    /// cache's identity for the texture's decoded words).
    pub fn bound(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        key: usize,
        texels: &std::sync::Arc<[[u8; 4]]>,
        width: u32,
        height: u32,
    ) -> Result<objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>, String> {
        use objc2_metal::{MTLStorageMode, MTLTexture as _, MTLTextureDescriptor, MTLTextureUsage};
        let expected = (width * height) as usize;
        let staged = self.textures.get(&key)
            .is_some_and(|pool| pool.width == width && pool.height == height);
        if !staged {
            let descriptor = MTLTextureDescriptor::new();
            descriptor.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
            unsafe {
                descriptor.setWidth(width as usize);
                descriptor.setHeight(height as usize);
            }
            descriptor.setUsage(MTLTextureUsage::ShaderRead);
            descriptor.setStorageMode(MTLStorageMode::Managed);
            let texture = device.newTextureWithDescriptor(&descriptor)
                .ok_or("the texture's staging did not allocate")?;
            self.textures.insert(key, PoolTexture { texture, width, height });
            let _ = expected;
        }
        let pool = self.textures.get_mut(&key).expect("the staged texture");
        // the texels' bytes: RGBA rows, uploaded whole when they change —
        // the same path the present's screens take (the replaceRegion's
        // writes at the decode's rate, which the cache already deduped)
        unsafe {
            pool.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                objc2_metal::MTLRegion {
                    origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize {
                        width: width as usize,
                        height: height as usize,
                        depth: 1,
                    },
                },
                0,
                std::ptr::NonNull::from(texels.as_flattened().as_ref()).cast(),
                (width * 4) as usize,
            );
        }
        Ok(pool.texture.clone())
    }
}

/// the words a batch gathers, staged once per batch: the Vulkan twin's
/// `Block` with the same dedupe (draws in a row mostly share the words —
/// the host rewrites only when the words differ).
#[derive(Default)]
pub struct Block {
    /// the words being gathered.
    words: Vec<u32>,
    /// the words staged last, and where, until the batch is handed over.
    last: Vec<u32>,
    at: Option<u64>,
}

impl Block {
    /// where the batch holds the words gathered, when they are the words
    /// the last block of the kind staged.
    pub fn staged(&self) -> Option<u64> {
        self.at.filter(|_| self.words == self.last)
    }

    /// the words gathered went into the batch at offset.
    pub fn staged_at(&mut self, offset: u64) {
        std::mem::swap(&mut self.words, &mut self.last);
        self.at = Some(offset);
    }
}

impl Ring {
    /// takes a span at the alignment asked, the Vulkan `stage`'s shape:
    /// `next_multiple_of` on the head, bounds against the capacity.
    pub fn stage(&mut self, size: u64, align: u64) -> Result<(u64, NonNull<u8>), String> {
        let offset = self.head.next_multiple_of(align.max(1));
        if offset + size > self.capacity {
            return Err(format!("a draw needs {size} more bytes than a batch has"));
        }
        self.head = offset + size;
        let extra = self.head - offset - size; // (the wrap while staging)
        let _ = extra;
        // reuse take's span bookkeeping for the aligned offset
        let contiguous = self
            .live
            .back()
            .is_some_and(|(start, bytes)| start + bytes == offset);
        if contiguous {
            // extend the last span
            let (start, bytes) = self.live.back_mut().unwrap();
            *bytes = offset + size - *start;
        } else {
            // not contiguous: its own span
            self.live.push_back((offset, size));
        }
        let pointer = unsafe {
            self.buffer.contents().cast::<u8>().add(offset as usize)
        };
        Ok((offset, pointer))
    }

    /// the write window for a staged span, the twin of hardware.rs's
    /// `ring(offset, size)`.
    pub fn span(&self, start: u64, size: u64) -> &mut [u8] {
        // SAFETY: the ring is Shared-mapped for its whole size and stage
        // kept the range inside it; the GPU is not reading this batch's
        // span while it is being recorded (A3b's one-frame channel says
        // the host and the GPU never touch the same bytes at once)
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.contents().cast::<u8>().add(start as usize).as_ptr(),
                size as usize,
            )
        }
    }

    /// stages the block's words once per batch, at the alignment asked —
    /// the Vulkan `stage_block`'s shape: the dedupe via Block::staged, the
    /// words written little-endian into the ring's bytes.
    pub fn stage_block(
        &mut self,
        block: &mut Block,
        align: u64,
    ) -> Result<u64, String> {
        let size = block.words.len() as u64 * 4;
        if let Some(at) = block.staged() {
            return Ok(at);
        }
        let (offset, _) = self.stage(size, align)?;
        let window = self.span(offset, size);
        for (out, word) in window.as_chunks_mut::<4>().0.iter_mut().zip(&block.words) {
            *out = word.to_le_bytes();
        }
        block.staged_at(offset);
        Ok(offset)
    }
}

pub(crate) use super::pica_state::draw_uniform_block;

/// One explicit texture/sampler pair; the caller owns filtering and wrapping.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[derive(Clone, Copy)]
pub struct TextureBinding<'a> {
    pub texture: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
    pub sampler: &'a ProtocolObject<dyn objc2_metal::MTLSamplerState>,
}

/// CPU-placed draw inputs, borrowed until synchronous GPU completion.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct PlacedDraw<'a> {
    /// the vertices' 24-float rows, packed the Vulkan `place`'s way.
    pub vertices: &'a [f32],
    /// three indices a triangle, the draw's triangles' shape.
    pub indices: &'a [u32],
    /// PICA texture units 0..2, with an explicit sampler for every bound unit.
    pub textures: [Option<TextureBinding<'a>>; 3],
    /// the render target's RGBA8 texture.
    pub target: &'a ProtocolObject<dyn objc2_metal::MTLTexture>,
    /// Explicit initialization only; subsequent draws must load both attachments.
    pub clear: bool,
    /// the depth+stencil twin, when the draw carries depth.
    pub depth: Option<&'a ProtocolObject<dyn objc2_metal::MTLTexture>>,
    /// the draw's viewport width and height (the vertices sit in them).
    pub width: u32,
    pub height: u32,
}

impl HardwareMetal {
    /// A draw's execution end to end: the uniforms and the vertices staged
    /// through the A4 ring, the pipeline built for the draw's depth shape,
    /// the textures bound by slot, the render pass encoded, committed and
    /// awaited. This is A4's draw executor — the app's `Resources` wire
    /// draws into it the way `resources.hardware` carries them into
    /// hardware.rs, and its honest remaining surface is the Vulkan twin's
    /// full-run way (surfaces from guest memory, the display transfers'
    /// destinations, the GPU-shading's compile race), carried the
    /// following chapter.
    pub fn draw_placed(
        &mut self,
        draw: &PlacedDraw<'_>,
        key: &PipelineKey,
        uniforms_words: &[u32],
        tables_words: &[u32],
    ) -> Result<(), String> {
        if key.vertex != VertexStage::Placed {
            return Err("Metal placed executor cannot run interpreted vertices".into());
        }
        if key.depth != draw.depth.is_some() {
            return Err("Metal depth attachment does not match pipeline key".into());
        }
        if uniforms_words.len() < super::pica_state::DRAW_WORDS {
            return Err("Metal Draw uniform block is truncated".into());
        }
        for unit in 0..3 {
            if uniforms_words[51] & (1 << unit) != 0 && draw.textures[unit].is_none() {
                return Err(format!("Metal texture unit {unit} is enabled but unbound"));
            }
        }
        let table_bytes = u32::try_from(tables_words.len() * 4)
            .map_err(|_| "Metal Tables buffer is too large")?;
        // the ring takes the uniforms, the tables, the vertices: the
        // Vulkan raster's batch's start
        let mut uniforms_block = Block { words: uniforms_words.to_vec(), last: Vec::new(), at: None };
        let uniforms_at = self.ring.stage_block(&mut uniforms_block, 16)?;
        let mut tables_block = Block { words: tables_words.to_vec(), last: Vec::new(), at: None };
        let tables_at = self.ring.stage_block(&mut tables_block, 16)?;

        let index_bytes = (draw.indices.len() * 4) as u64;
        let (index_at, _) = self.ring.stage(index_bytes, 4)?;
        let staging = self.ring.span(index_at, index_bytes);
        for (out, value) in staging.as_chunks_mut::<4>().0.iter_mut().zip(draw.indices) {
            *out = value.to_le_bytes();
        }
        let vertex_bytes = (draw.vertices.len() * 4) as u64;
        let (vertex_at, _) = self.ring.stage(vertex_bytes, 16)?;
        let staging = self.ring.span(vertex_at, vertex_bytes);
        for (bytes, value) in staging.as_chunks_mut::<4>().0.iter_mut().zip(draw.vertices) {
            *bytes = value.to_le_bytes();
        }

        // the batches' route: the vertices in buffer 0, the uniforms in 3,
        // the tables in 4, and the pipeline the draw's key builds
        let pipeline = self.pipeline_for_key(key)?;

        let pass = objc2_metal::MTLRenderPassDescriptor::new();
        unsafe {
            let attachments = pass.colorAttachments();
            let colour: Retained<objc2_metal::MTLRenderPassColorAttachmentDescriptor> =
                attachments.objectAtIndexedSubscript(0);
            colour.setTexture(Some(draw.target));
            colour.setLoadAction(if draw.clear { MTLLoadAction::Clear } else { MTLLoadAction::Load });
            colour.setStoreAction(MTLStoreAction::Store);
            colour.setClearColor(MTLClearColor { red: 0.04, green: 0.05, blue: 0.09, alpha: 1.0 });
            // the depth's twin, when the draw carries one
            if let Some(depth) = draw.depth {
                let depth_attachment = pass.depthAttachment();
                depth_attachment.setTexture(Some(depth));
                depth_attachment.setLoadAction(if draw.clear { MTLLoadAction::Clear } else { MTLLoadAction::Load });
                depth_attachment.setStoreAction(MTLStoreAction::Store);
                depth_attachment.setClearDepth(1.0);
            }
        }
        let buffer = self.queue.commandBuffer().expect("the command buffer");
        let encoder = unsafe { buffer.renderCommandEncoderWithDescriptor(&pass) }
            .expect("the draw's render encoder");
        unsafe {
            encoder.setRenderPipelineState(&pipeline);
            encoder.setVertexBuffer_offset_atIndex(Some(self.ring.buffer_of()), vertex_at as usize, 0);
            encoder.setFragmentBuffer_offset_atIndex(Some(self.ring.buffer_of()), uniforms_at as usize, 3);
            encoder.setFragmentBuffer_offset_atIndex(Some(self.ring.buffer_of()), tables_at as usize, 4);
            for (unit, binding) in draw.textures.iter().enumerate() {
                encoder.setFragmentTexture_atIndex(binding.map(|b| b.texture), unit);
                encoder.setFragmentSamplerState_atIndex(binding.map(|b| b.sampler), 10 + unit);
            }
            // Naga's only runtime-sized global is Tables: one byte-length word.
            encoder.setFragmentBytes_length_atIndex(
                NonNull::from(&table_bytes).cast(), 4, 30);
            encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                objc2_metal::MTLPrimitiveType::Triangle,
                draw.indices.len(),
                objc2_metal::MTLIndexType::UInt32,
                self.ring.buffer_of(),
                index_at as usize,
            );
            let encoder_ref: &AnyObject = encoder.as_ref();
            let _: () = objc2::msg_send![encoder_ref, endEncoding];
        }
        buffer.commit();
        buffer.waitUntilCompleted();
        Ok(())
    }

    /// the pipeline per key: the DYNAMIC path's rack shaders, the key's
    /// fragment words baked through `prepare`, the vertex placed or shaded.
    fn pipeline_for_key(
        &mut self,
        key: &PipelineKey,
    ) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>>, String> {
        use objc2_metal::{
            MTLRenderPipelineDescriptor, MTLVertexDescriptor, MTLVertexFormat,
            MTLVertexStepFunction, MTLTexture as _,
        };
        // the fragment: the KEY's words baked in through prepare, the
        // bindings the fragment's own per-entry map
        let mut fragment_bindings = naga_msl::BindingMap::default();
        for binding in 0..3u32 {
            fragment_bindings.insert(
                naga::ResourceBinding { group: 0, binding },
                texture_target(binding as u8),
            );
        }
        fragment_bindings.insert(
            naga::ResourceBinding { group: 0, binding: 3 },
            buffer_target(3),
        );
        fragment_bindings.insert(
            naga::ResourceBinding { group: 0, binding: 4 },
            buffer_target(4),
        );
        fragment_bindings.insert(
            naga::ResourceBinding { group: 0, binding: 24 },
            buffer_target(24),
        );
        for slot in 10..13u32 {
            fragment_bindings.insert(
                naga::ResourceBinding { group: 0, binding: slot },
                sampler_target(slot as u8),
            );
        }
        let mut per_entry = naga_msl::EntryPointResourceMap::default();
        per_entry.insert(String::from("main"), naga_msl::EntryPointResources {
            resources: fragment_bindings,
            sizes_buffer: Some(30),
            ..Default::default()
        });

        let (fragment_msl, fragment_entry) = glsl::compile_msl_stage_entry(
            &prepare(RASTER_FRAG, Some(&key_fragment(key)), false),
            ShaderStage::Fragment,
            &[],
            Some(&per_entry),
        )?;
        let (vertex_msl, vertex_entry) = glsl::compile_msl_stage_entry(
            &prepare(RASTER_VERT, None, false),
            ShaderStage::Vertex,
            &[],
            None,
        ).expect("the vertex translates");
        let fragment_library = self.device
            .newLibraryWithSource_options_error(&NSString::from_str(&fragment_msl), None)
            .map_err(|error| format!("the keyed fragment MSL: {error}"))?;
        let vertex_library = self.device
            .newLibraryWithSource_options_error(&NSString::from_str(&vertex_msl), None)
            .map_err(|error| format!("the vertex MSL: {error}"))?;
        let vertex_function = vertex_library.newFunctionWithName(&ns(&vertex_entry))
            .expect("the vertex main");
        let fragment_function = fragment_library.newFunctionWithName(&ns(&fragment_entry))
            .expect("the fragment main");

        let descriptor = MTLRenderPipelineDescriptor::new();
        let depth_format = if key.depth {
            depth_pixel_format(key.float_depth)
        } else {
            // without depth the descriptor only needs the colour target
            MTLPixelFormat::Invalid
        };
        descriptor.setStencilAttachmentPixelFormat(depth_format);
        descriptor.setDepthAttachmentPixelFormat(depth_format);
        unsafe {
            let attachments = descriptor.colorAttachments();
            let colour: Retained<objc2_metal::MTLRenderPipelineColorAttachmentDescriptor> =
                attachments.objectAtIndexedSubscript(0);
            colour.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
            colour.setWriteMask(metal_color_mask(key.mask));
        }
        descriptor.setRasterSampleCount(1);
        descriptor.setVertexFunction(Some(vertex_function.as_ref()));
        descriptor.setFragmentFunction(Some(fragment_function.as_ref()));
        let vertex_descriptor: Retained<MTLVertexDescriptor> =
            MTLVertexDescriptor::vertexDescriptor();
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
            layout.setStride(96);
            layout.setStepFunction(MTLVertexStepFunction::PerVertex);
            descriptor.setVertexDescriptor(Some(&vertex_descriptor));
        }
        unsafe { self.device.newRenderPipelineStateWithDescriptor_error(&descriptor) }
            .map_err(|error| format!("no pipeline state: {error}"))
    }
}

/// PICA RGBA bits are low-to-high; Metal defines red at bit 3.
fn metal_color_mask(mask: u32) -> objc2_metal::MTLColorWriteMask {
    use objc2_metal::MTLColorWriteMask as Mask;
    let mut result = Mask::empty();
    for (bit, channel) in [Mask::Red, Mask::Green, Mask::Blue, Mask::Alpha].into_iter().enumerate() {
        if mask & (1 << bit) != 0 { result |= channel; }
    }
    result
}

fn sampler_target(slot: u8) -> naga_msl::BindTarget {
    naga_msl::BindTarget {
        sampler: Some(naga_msl::BindSamplerTarget::Resource(slot)),
        ..Default::default()
    }
}

/// the fragment words a key's shape bakes: DYNAMIC picks the generic path,
/// the combination words ride the draw's uniforms when it is set.
fn key_fragment(key: &PipelineKey) -> [u32; FRAGMENT_CONSTANTS] {
    let _ = key;
    let mut values = [0u32; FRAGMENT_CONSTANTS];
    values[29] = 1; // DYNAMIC: the specials carry the draw's own
    values
}

/// The A4's surfaces: the draw targets backed by the guest's memory, the
/// honest simplified core of the Vulkan raster's `Surface` — a Managed
/// RGBA8 texture at the console's resolution, the guest's shadow, and the
/// rows' load-then-draw-then-writeback. The Vulkan twin's `native` scaled
/// images, the guards, the stale rows and the twin (frame interpolation)
/// land after the surfaces run.
#[cfg(all(target_os = "macos", feature = "metal"))]
struct MetalSurface {
    texture: Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>,
    addr: u32,
    width: u32,
    height: u32,
    format: ColorFormat,
    tiled: bool,
    /// the guest's bytes the texture matched last.
    shadow: Vec<u8>,
}

impl MetalSurface {
    /// the guest's bytes a whole surface takes.
    fn size(&self) -> u32 {
        self.width * self.height * self.format.bytes_per_pixel() as u32
    }

    /// the bytes one row of the guest's buffer takes.
    fn row_bytes(&self) -> u32 {
        self.width * self.format.bytes_per_pixel() as u32
    }

    /// whether the texture's bytes match the guest's (the honest shadow,
    /// bytes compared where the memory is in one piece).
    fn shadows(&self, bytes: &[u8]) -> bool {
        !self.shadow.is_empty() && bytes == self.shadow
    }

    /// the GPU drew: record the shadow from the texture's rows.
    fn keep(&mut self, bytes: &[u8]) {
        self.shadow = bytes.to_vec();
    }
}

impl HardwareMetal {
    /// the GPU drew into the surface: its rows come back to the guest,
    /// re-encoded through the surface's own format (the shared `encode`,
    /// the guest's ABGR memory order), the shadow carrying what memory
    /// holds now — the Vulkan `store`'s shape through a Managed read-back.
    fn write_back<M: crate::GpuMemory>(
        &mut self,
        memory: &mut M,
        index: usize,
    ) -> Result<(), String> {
        use objc2_metal::MTLTexture as _;
        let texture = self.surfaces[index].texture.clone();
        let (width, height, format, tiled) = {
            let surface = &self.surfaces[index];
            (surface.width, surface.height, surface.format, surface.tiled)
        };
        let bpp = format.bytes_per_pixel();
        // the RGBA read-back rows top-first
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::from(rgba.as_slice()).cast(),
                (width * 4) as usize,
                objc2_metal::MTLRegion {
                    origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize {
                        width: width as usize, height: height as usize, depth: 1,
                    },
                },
                0,
            );
        }
        // the guest's bytes: rows bottom-first, the pixels re-encoded
        let mut bytes = vec![0u8; self.surfaces[index].size() as usize];
        for y in 0..height {
            for x in 0..width {
                let at = (((height - 1 - y) * width + x) * 4) as usize;
                let out = if tiled {
                    crate::format::morton_offset(x, y, width, bpp as u32) as usize
                } else {
                    (y as usize * width as usize + x as usize) * bpp
                };
                format.encode(
                    [rgba[at], rgba[at + 1], rgba[at + 2], rgba[at + 3]],
                    &mut bytes[out..out + bpp],
                );
            }
        }
        memory.write(self.surfaces[index].addr, &bytes);
        self.surfaces[index].keep(&bytes);
        Ok(())
    }

    /// looks the surface up or makes it from the guest's rows, uploading
    /// when the bytes changed — the Vulkan `surface_rows`'s shape with the
    /// CPU decode (the shared `format.decode` + morton), the upload into
    /// the texture's flipped rows (the guest's rows run bottom first).
    fn check_surface<M: crate::GpuMemory>(
        &mut self,
        memory: &mut M,
        addr: u32,
        (width, height): (u32, u32),
        format: ColorFormat,
        tiled: bool,
    ) -> Result<usize, String> {
        use objc2_metal::{MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage};
        let index = self
            .surfaces
            .iter()
            .position(|surface| {
                surface.addr == addr && surface.width == width
                    && surface.height == height && surface.format == format
                    && surface.tiled == tiled
            });
        let index = match index {
            Some(index) => index,
            None => {
                let descriptor = MTLTextureDescriptor::new();
                descriptor.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
                unsafe {
                    descriptor.setWidth(width as usize);
                    descriptor.setHeight(height as usize);
                }
                descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
                descriptor.setStorageMode(MTLStorageMode::Managed);
                let texture = self.device.newTextureWithDescriptor(&descriptor)
                    .ok_or("the surface's texture did not allocate")?;
                self.surfaces.push(MetalSurface {
                    texture,
                    addr,
                    width,
                    height,
                    format,
                    tiled,
                    shadow: Vec::new(),
                });
                self.surfaces.len() - 1
            }
        };

        // the shadow: the bytes when the texture matched. changed, they get
        // read and decoded into the texture's flipped rows
        let size = self.surfaces[index].size() as usize;
        let mut bytes = vec![0u8; size];
        memory.read(addr, &mut bytes);
        if !self.surfaces[index].shadows(&bytes) {
            let texture = &self.surfaces[index];
            let (w, h, bpp) = (
                texture.width,
                texture.height,
                texture.format.bytes_per_pixel() as u32,
            );
            let mut texture_bytes = vec![0u8; (w * h * 4) as usize];
            let decode = texture.format;
            let (w_us, h_us, bpp_us) = (w as usize, h as usize, bpp as usize);
            for y in 0..h {
                for x in 0..w {
                    let at = if texture.tiled {
                        crate::format::morton_offset(x, y, w, bpp) as usize
                    } else {
                        ((y * w + x) * bpp) as usize
                    };
                    let rgba: [u8; 4] = decode.decode(&bytes[at..at + bpp_us]);
                    let out = ((h_us - 1 - y as usize) * w_us + x as usize) * 4;
                    texture_bytes[out..out + 4].copy_from_slice(&rgba);
                }
            }
            unsafe {
                use objc2_metal::MTLTexture as _;
                self.surfaces[index].texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    objc2_metal::MTLRegion {
                        origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                        size: objc2_metal::MTLSize { width: w as usize, height: h as usize, depth: 1 },
                    },
                    0,
                    std::ptr::NonNull::from(texture_bytes.as_slice()).cast(),
                    (w * 4) as usize,
                );
            }
            self.surfaces[index].keep(&bytes);
        }
        Ok(index)
    }
}

/// the display-transfer's run on the Metal host: the input surface loads
/// from the guest (the check_surface's honest load), the output surface
/// mounts, and the transfer's compute shader averages the game's pixels
/// through their own formats. the push constants carry the shader's words:
/// size, input height, output height, scale, flip, and the two formats.
#[cfg(all(target_os = "macos", feature = "metal"))]
impl HardwareMetal {
    pub(crate) fn display_transfer<M: crate::GpuMemory>(
        &mut self,
        memory: &mut M,
        transfer: &super::Transfer,
    ) -> Result<(), String> {
        use objc2_metal::{MTLCommandBuffer as _, MTLComputeCommandEncoder as _, MTLTexture as _};
        use objc2::runtime::AnyObject;

        // the input surface from the guest: its own height carries the
        // rows the scale reads (the transfer's blocks read size.y * scale.y)
        let input_height = transfer.copy.1 * transfer.scale.1;
        let input_index = self.check_surface(
            memory,
            transfer.input,
            (transfer.input_width, input_height),
            transfer.input_format,
            !transfer.input_linear,
        )?;
        let input = self.surfaces[input_index].texture.clone();

        // the output surface: the target's size at the output's format —
        // the output's tiles are plain rows if the transfer leaves them so
        let output_index = self.check_surface(
            memory,
            transfer.output,
            (transfer.output_width, transfer.output_height),
            transfer.output_format,
            transfer.output_tiled,
        )?;

        // the compute: once per the rack's translation
        let rack = rack()?;
        let compute_library = self.device
            .newLibraryWithSource_options_error(&NSString::from_str(&rack.transfer_comp.msl), None)
            .map_err(|error| format!("the transfer's MSL: {error}"))?;
        let kernel = compute_library.newFunctionWithName(&ns(&rack.transfer_comp.entry))
            .expect("the transfer main");
        let compute = unsafe {
            self.device.newComputePipelineStateWithFunction_error(kernel.as_ref())
        }.map_err(|error| format!("the transfer compute pipeline: {error}"))?;

        // the push constants: the shader's words in the GLSL block's order
        // — the honest surfaces run at the console's resolution (scale 1)
        let mut push = [0u32; 11];
        push[0] = transfer.copy.0;                 // size.x: the output's width
        push[1] = transfer.copy.1;                 // size.y
        push[2] = input_height;                    // the input's own height
        push[3] = transfer.output_height;          // the output's height
        push[4] = transfer.scale.0;
        push[5] = transfer.scale.1;
        push[6] = transfer.flip as u32;
        push[7] = transfer.input_format as u32;
        push[8] = transfer.output_format as u32;
        push[9] = 0;                               // the input starts at its row 0
        push[10] = input_height;
        let push_buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::from(push.as_ref()).cast(),
                push.len() * 4,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
        }.expect("the transfer's push buffer");

        let buffer = self.queue.commandBuffer().expect("the command buffer");
        let encoder = unsafe { buffer.computeCommandEncoder() }.expect("the compute encoder");
        unsafe {
            encoder.setComputePipelineState(&compute);
            // the input's image is the surface's own texture at slot 0
            encoder.setTexture_atIndex(Some(input.as_ref()), 0);
            let output = self.surfaces[output_index].texture.clone();
            encoder.setTexture_atIndex(Some(output.as_ref()), 1);
            encoder.setBuffer_offset_atIndex(Some(push_buffer.as_ref()), 0, 24);
            let threads = objc2_metal::MTLSize {
                width: transfer.copy.0 as usize, height: transfer.copy.1 as usize, depth: 1,
            };
            let group = objc2_metal::MTLSize {
                width: 1, height: 1, depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(group, threads);
            let encoder_ref: &AnyObject = encoder.as_ref();
            let _: () = objc2::msg_send![encoder_ref, endEncoding];
        }
        buffer.commit();
        buffer.waitUntilCompleted();
        Ok(())
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
mod tests {
    #[test]
    fn pica_color_masks_map_all_channel_combinations_to_metal() {
        use objc2_metal::MTLColorWriteMask as Mask;
        let channels = [Mask::Red, Mask::Green, Mask::Blue, Mask::Alpha];
        for bits in 0..16u32 {
            let actual = super::metal_color_mask(bits);
            for (index, channel) in channels.into_iter().enumerate() {
                assert_eq!(actual.contains(channel), bits & (1 << index) != 0);
            }
        }
    }

    use super::*;
    use objc2_metal::{MTLDevice as _, MTLPixelFormat};

    /// a small guest memory the tests write and read through the trait's
    /// shape (the ConsoleMemory's watch/unchanged land with A4's full
    /// surfaces batch work).
    struct HostMemory {
        bytes: std::collections::HashMap<u64, Vec<u8>>,
    }
    impl HostMemory {
        fn new() -> Self {
            Self { bytes: std::collections::HashMap::new() }
        }
        fn write_raw(&mut self, addr: u64, bytes: &[u8]) {
            self.bytes.insert(addr, bytes.to_vec());
        }
    }
    impl crate::GpuMemory for HostMemory {
        fn read(&mut self, addr: u32, buf: &mut [u8]) {
            let addr = addr as u64;
            for (index, byte) in buf.iter_mut().enumerate() {
                *byte = self.bytes.get(&addr).and_then(|run| run.get(index).copied()).unwrap_or(0);
            }
        }
        fn write(&mut self, addr: u32, data: &[u8]) {
            self.bytes.insert(addr as u64, data.to_vec());
        }
    }

    #[test]
    fn the_ring_allocates_flushes_and_retires() {
        use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice as _};
        let device = MTLCreateSystemDefaultDevice().expect("the device");
        let mut ring = Ring::new(device.as_ref(), 1 << 12).expect("the ring");
        assert_eq!(ring.free(), 1 << 12);
        // empty ring never wants a flush
        assert!(!ring.should_flush());

        // two spans, no wrap: the head climbs
        let (start0, pointer) = ring.take(256).expect("the span 0");
        let (start1, _) = ring.take(512).expect("the span 1");
        assert_eq!(start0, 0);
        assert_eq!(start1, 256);
        // the shared bytes' pointer sits the span's offset into the ring
        // the pointer sits at the ring's content base (the span's start 0)
        unsafe {
            use objc2_metal::MTLBuffer as _;
            let base = ring.buffer_of().contents();
            assert_eq!(pointer.as_ptr(), base.add(0).cast::<u8>().as_ptr());
        }
        assert_eq!(ring.free(), (1 << 12) - 768);

        // the spans retire once their command buffer completed
        ring.retire(start0, 256);
        ring.retire(start1, 512);
        assert_eq!(ring.free(), 1 << 12, "everything retires back");

        // a span past the tall tail wraps the head to the start
        let (start2, _) = ring.take(3400).expect("the span wraps");
        assert_eq!(start2, 0, "the wrapped span starts over");
        assert!(ring.should_flush(), "half the ring walked: the flush fires");

        // taking too much errors honestly
        let too_big = ring.take((1 << 12) as u64);
        assert!(too_big.is_err(), "the overflow refuses");
    }

    #[test]
    fn surfaces_load_from_guest_memory_when_the_bytes_change() {
        use crate::GpuMemory;

        let mut hardware = HardwareMetal::new().expect("the Metal raster constructs");
        let mut memory = HostMemory::new();

        // the guest's tiled RGBA8 rows: bottom-first rows, each pixel's
        // value the formula (x, 255, 0, 255)
        const SIZE: u32 = 32;
        let buffer_addr: u32 = 0x0100_0000;
        let mut bytes = vec![0u8; (SIZE * SIZE * 4) as usize];
        for y in 0..SIZE {
            for x in 0..SIZE {
                let at = crate::format::morton_offset(x, y, SIZE, 4) as usize;
                bytes[at] = (x * 8) as u8; bytes[at + 1] = 255; bytes[at + 2] = 0; bytes[at + 3] = 255;
            }
        }
        memory.write_raw(buffer_addr as u64, &bytes);

        // the first check: the surface loads the guest's rows
        let index = hardware.check_surface(&mut memory, buffer_addr, (SIZE, SIZE), ColorFormat::Rgba8, true)
            .expect("the surface checks");
        assert_eq!(index, 0, "the first surface lands alone");
        assert_eq!(hardware.surfaces[index].shadow, bytes, "the shadow carries the guest's");

        // the same bytes: the shadow holds, no upload lands
        let index_again = hardware.check_surface(&mut memory, buffer_addr, (SIZE, SIZE), ColorFormat::Rgba8, true)
            .expect("the surface re-checks");
        assert_eq!(index_again, index);

        // the guest changes a pixel: the whole rows rewrite (the fake's
        // runs sit whole), the next check uploads again
        let change = crate::format::morton_offset(10, 5, SIZE, 4) as usize;
        bytes[change] = 255; bytes[change + 1] = 255; bytes[change + 3] = 255;
        memory.write_raw(buffer_addr as u64, &bytes);
        hardware.check_surface(&mut memory, buffer_addr, (SIZE, SIZE), ColorFormat::Rgba8, true)
            .expect("the changed surface re-checks");
        assert_eq!(hardware.surfaces[index].shadow[change], 255u8, "the shadow follows the guest");
    }

    #[test]
    fn pixel_formats_match_the_vulkan_twins() {
        use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice as _, MTLTexture as _};
        use std::sync::Arc;
        let device = MTLCreateSystemDefaultDevice().expect("the device");
        let mut pool = TexturePool::new();

        let first: Arc<[[u8; 4]]> = Arc::from(vec![[0u8, 255, 0, 255]; 4 * 4]);
        let bound = pool.bound(device.as_ref(), 0x1000, &first, 4, 4)
            .expect("the first texture binds");
        assert!(bound.width() == 4, "the texture carries its width");

        // the same texels' identity: the same texture object returns
        let again = pool.bound(device.as_ref(), 0x1000, &first, 4, 4)
            .expect("the dedupe binds");
        assert_eq!(
            Retained::as_ptr(&bound) as usize,
            Retained::as_ptr(&again) as usize,
            "the identity keeps the texture"
        );

        // a different decode: a different texture
        let second: Arc<[[u8; 4]]> = Arc::from(vec![[255u8, 0, 0, 255]; 8 * 8]);
        let other = pool.bound(device.as_ref(), 0x2000, &second, 8, 8)
            .expect("the second texture binds");
        assert_ne!(
            Retained::as_ptr(&bound) as usize,
            Retained::as_ptr(&other) as usize,
            "the texels' new identity binds fresh"
        );
    }

    #[test]
    fn stage_block_stages_once_and_reuses_the_offset() {
        use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice as _};
        let device = MTLCreateSystemDefaultDevice().expect("the device");
        let mut ring = Ring::new(device.as_ref(), 1 << 12).expect("the ring");
        let mut block = Block::default();
        block.words = vec![0xDEAD_BEEF, 0x1234_5678];

        // first stage: the span lands at the 16-byte alignment the draw
        // uniforms ask, the words land little-endian in the bytes
        let first = ring.stage_block(&mut block, 16).expect("the first stage");
        assert_eq!(first, 0, "the aligned head starts the batch");
        let bytes = ring.span(first, 8);
        assert_eq!(bytes, &[0xEF, 0xBE, 0xAD, 0xDE, 0x78, 0x56, 0x34, 0x12]);

        // the shared draw re-gathers the same words: the dedupe returns
        // the offset bygones (the words empty after the swap, the draw
        // refills them identically — the Vulkan Block's own shape)
        block.words = vec![0xDEAD_BEEF, 0x1234_5678];
        let again = ring.stage_block(&mut block, 16).expect("the dedupe");
        assert_eq!(again, first, "the shared words restage for free");

        // new words: the stage walks the head
        block.words = vec![1, 2, 3, 4];
        let third = ring.stage_block(&mut block, 16).expect("the next stage");
        assert!(third >= first + 8, "the new words stage past the old");
        let bytes = ring.span(third, 16);
        assert_eq!(bytes, &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]);
    }

    #[test]
    fn default_pipelines_build_on_the_device() {
        let hardware = HardwareMetal::new().expect("the Metal raster constructs");
        // the float twin is what Apple silicon renders (the D24 rejects)
        for (depth, float_depth) in [(true, true), (false, true)] {
            let pipeline = hardware
                .graphics_pipeline(
                    "raster.vert",
                    "raster.frag",
                    &PipelineKey { mask: 0xF, depth, float_depth, vertex: VertexStage::Placed },
                )
                .expect("the default pipelines construct");
            let _ = pipeline;
        }
        // every rack shader hands back its function
        for shader in [
            "shade.vert", "raster.vert", "raster.frag", "raster_depth.frag",
            "transfer.comp", "depth.comp", "upright.comp", "upright_image.comp",
        ] {
            hardware.function(shader).expect(shader);
        }
    }

    /// A3a's execution proof: one command queue draw into an off-screen
    /// target through the whole chain the raster uses — translation, a
    /// pipeline, vertex buffers, the encoder, commit, a GPU-side wait, and
    /// pixels read back. The test shaders are this module's own (the PICA
    /// fragment semantics belong to A3b's uniform wiring); the mechanism
    /// this lands IS the landing itself.
    #[test]
    fn a_draw_runs_through_the_pipeline_to_pixels() {
        use objc2_metal::{
            MTLClearColor, MTLLoadAction, MTLPrimitiveType, MTLRegion, MTLOrigin,
            MTLRenderPipelineDescriptor, MTLRenderPassDescriptor, MTLResourceOptions,
            MTLStorageMode, MTLStoreAction, MTLTextureDescriptor, MTLTextureUsage,
            MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction,
            MTLCommandBuffer as _, MTLTexture as _, MTLDevice as _,
            MTLRenderCommandEncoder as _, MTLRenderPipelineState as _,
        };
        use objc2::runtime::AnyObject;
        use std::ptr::NonNull;

        let (vertex_msl, vertex_entry) = glsl::compile_msl_stage_entry(
            &tri_vertex(), ShaderStage::Vertex, &[], None,
        ).expect("the test vertex shader translates");
        let (fragment_msl, fragment_entry) = glsl::compile_msl_stage_entry(
            &tri_fragment(), ShaderStage::Fragment, &[], None,
        ).expect("the test fragment shader translates");

        let hardware = HardwareMetal::new().expect("the Metal raster constructs");
        let vertex_library = hardware.device
            .newLibraryWithSource_options_error(&NSString::from_str(&vertex_msl), None)
            .expect("the test vertex MSL compiles");
        let fragment_library = hardware.device
            .newLibraryWithSource_options_error(&NSString::from_str(&fragment_msl), None)
            .expect("the test fragment MSL compiles");
        let vertex_function = vertex_library.newFunctionWithName(&ns(&vertex_entry))
            .expect("the vertex main");
        let fragment_function = fragment_library.newFunctionWithName(&ns(&fragment_entry))
            .expect("the fragment main");

        // the pipeline: no depth attachment, the color RGBA8; the vertex
        // slots carry the Float3 position and the Float4 color in slot 0
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(vertex_function.as_ref()));
        descriptor.setFragmentFunction(Some(fragment_function.as_ref()));
        unsafe {
            let attachments = descriptor.colorAttachments();
            let color: Retained<objc2_metal::MTLRenderPipelineColorAttachmentDescriptor> =
                attachments.objectAtIndexedSubscript(0);
            color.setPixelFormat(MTLPixelFormat::RGBA8Unorm);

            let vertex_descriptor: Retained<MTLVertexDescriptor> =
                MTLVertexDescriptor::vertexDescriptor();
            let attributes = vertex_descriptor.attributes();
            let position: Retained<objc2_metal::MTLVertexAttributeDescriptor> =
                attributes.objectAtIndexedSubscript(0);
            position.setFormat(MTLVertexFormat::Float3);
            position.setOffset(0);
            position.setBufferIndex(0);
            let painted: Retained<objc2_metal::MTLVertexAttributeDescriptor> =
                attributes.objectAtIndexedSubscript(1);
            painted.setFormat(MTLVertexFormat::Float4);
            painted.setOffset(12);
            painted.setBufferIndex(0);
            let layouts = vertex_descriptor.layouts();
            let layout: Retained<objc2_metal::MTLVertexBufferLayoutDescriptor> =
                layouts.objectAtIndexedSubscript(0);
            layout.setStride(28);
            layout.setStepFunction(MTLVertexStepFunction::PerVertex);
            descriptor.setVertexDescriptor(Some(&vertex_descriptor));
        }
        let spinner = unsafe {
            hardware.device.newRenderPipelineStateWithDescriptor_error(&descriptor)
        }.expect("the test pipeline constructs");

        // 28 bytes a vertex, a green triangle over the presenter's navy
        let vertices: [f32; 21] = [
            //   x     y    z     r    g    b    a
            -1.0, -1.0, 0.5,   0.0, 1.0, 0.0, 1.0,
             1.0, -1.0, 0.5,   0.0, 1.0, 0.0, 1.0,
            -1.0,  1.0, 0.5,   0.0, 1.0, 0.0, 1.0,
        ];
        let vertex_buffer = unsafe {
            hardware.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::from(vertices.as_slice()).cast(),
                vertices.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }.expect("the vertex buffer allocates");

        // the target texture, Managed so the CPU can read its pixels
        const WIDTH: usize = 32;
        const HEIGHT: usize = 32;
        let target_descriptor = MTLTextureDescriptor::new();
        target_descriptor.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        unsafe {
            target_descriptor.setWidth(WIDTH);
            target_descriptor.setHeight(HEIGHT);
        }
        target_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        target_descriptor.setStorageMode(MTLStorageMode::Managed);
        let target = hardware.device
            .newTextureWithDescriptor(&target_descriptor)
            .expect("the render target allocates");

        // the frame: clear to the presenter's letterbox navy, draw green
        let pass = MTLRenderPassDescriptor::new();
        unsafe {
            let attachments = pass.colorAttachments();
            let attachment: Retained<objc2_metal::MTLRenderPassColorAttachmentDescriptor> =
                attachments.objectAtIndexedSubscript(0);
            attachment.setTexture(Some(target.as_ref()));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setStoreAction(MTLStoreAction::Store);
            attachment.setClearColor(MTLClearColor {
                red: 0.04, green: 0.05, blue: 0.09, alpha: 1.0,
            });
        }
        let buffer = hardware.queue.commandBuffer().expect("the command buffer");
        let encoder = unsafe { buffer.renderCommandEncoderWithDescriptor(&pass) }
            .expect("the render encoder");
        unsafe {
            encoder.setRenderPipelineState(&spinner);
            encoder.setVertexBuffer_offset_atIndex(Some(vertex_buffer.as_ref()), 0, 0);
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            let encoder_ref: &AnyObject = encoder.as_ref();
            let _: () = objc2::msg_send![encoder_ref, endEncoding];
        }
        buffer.commit();
        buffer.waitUntilCompleted();

        // read the pixels back and prove the triangle drew
        let mut pixels = [0u8; WIDTH * HEIGHT * 4];
        unsafe {
            target.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::from(pixels.as_mut_slice()).cast(),
                WIDTH * 4,
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize { width: WIDTH, height: HEIGHT, depth: 1 },
                },
                0,
            );
        }

        // the raster draws green in the triangle, and only there
        let screen = |x: usize, y: usize| {
            let at = (y * WIDTH + x) * 4;
            [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
        };
        let green = [0u8, 255, 0, 255];
        let navy = [10u8, 13, 23, 255];
        assert_eq!(screen(3, 28), green, "the screen's lower-left inside the triangle");
        assert_eq!(screen(1, 2), green, "just past the top-left tip");
        assert_eq!(screen(28, 2), navy, "the top-right is outside");
        assert_eq!(screen(28, 28), navy, "the bottom-right is outside");
    }

    #[test]
    fn the_write_back_carries_the_drawn_pixels_to_memory() {
        use objc2_metal::{MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage, MTLTexture as _};

        let mut hardware = HardwareMetal::new().expect("the Metal raster constructs");
        let mut memory = HostMemory::new();
        const SIZE: u32 = 32;
        let buffer_addr: u32 = 0x0100_0000;

        // a plain surface made direct (the texture's own RGBA8): the rows
        // empty — the draw paints green into the texture and the write-back
        // carries it to the guest
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        unsafe {
            descriptor.setWidth(SIZE as usize);
            descriptor.setHeight(SIZE as usize);
        }
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Managed);
        let texture = hardware.device.newTextureWithDescriptor(&descriptor)
            .expect("the surface's texture");
        hardware.surfaces.push(MetalSurface {
            texture,
            addr: buffer_addr,
            width: SIZE, height: SIZE,
            format: ColorFormat::Rgba8,
            tiled: false,
            shadow: Vec::new(),
        });

        // the write-back: the empty texture encodes zeros into the guest
        hardware.write_back(&mut memory, 0).expect("the write-back");
        // the plain surface's bytes read linear: the pixel (5, 10)'s own
        // words sit at ((10*32)+5)*4, the ABGR order the encode carries
        let at = ((10 * 32 + 5) * 4) as usize;

        // the draw paints a green pixel directly in the surface's texture
        let green: [u8; 4] = [0u8, 255, 0, 255]; // memory's rgba rows stay rgba
        unsafe {
            use objc2_metal::MTLTexture as _;
            let texture = &hardware.surfaces[0].texture;
            // the texture's rows run top-first: the guest's (5, 10) sits at
            // the texture's row (height-1-10) — the write-back's own map
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                objc2_metal::MTLRegion {
                    origin: objc2_metal::MTLOrigin { x: 5, y: SIZE as usize - 1 - 10, z: 0 },
                    size: objc2_metal::MTLSize { width: 1, height: 1, depth: 1 },
                },
                0,
                std::ptr::NonNull::from(&green[..]).cast(),
                4,
            );
        }

        // the write-back again: the guest reads the drawn pixel back —
        // the ABGR words: mem0 = a (255), mem2 = g (255), mem3 = r (0)
        hardware.write_back(&mut memory, 0).expect("the write-back again");
        let alpha = memory.read_entry_byte(buffer_addr as u64 + at as u64)
            .expect("the guest's alpha byte");
        assert_eq!(alpha, 255u8, "the pixel's alpha full");
        let green_read = memory.read_entry_byte(buffer_addr as u64 + at as u64 + 2)
            .expect("the green byte");
        assert_eq!(green_read, 255u8, "the drawn green carried to memory");
        let red_read = memory.read_entry_byte(buffer_addr as u64 + at as u64 + 3)
            .expect("the red byte");
        assert_eq!(red_read, 0u8, "the pixel's red zero");
    }

    /// HostMemory gains the byte-level reads the write-back's test needs.
    impl HostMemory {
        fn read_entry_byte(&self, addr: u64) -> Option<u8> {
            // the byte lives inside a run that covers it (the write-back
            // stores the whole surface under the surface's own base)
            for (base, run) in &self.bytes {
                if addr >= *base && addr < *base + run.len() as u64 {
                    return run.get((addr - *base) as usize).copied();
                }
            }
            None
        }
    }

    /// the raster's real fragment semantics over the draw block — the
    /// DYNAMIC path reads tev/misc/units from the uniforms, the combiner
    /// stages hand the vertex colour straight through, and the resource
    /// bindings map to the Vulkan bindings' numbers (textures 0..2, the
    /// buffer 3 the draw block, 4 the tables, samplers 10..12).
    #[test]
    fn the_raster_fragment_reads_the_draw_block() {
        use naga::back::msl as naga_msl;
        use objc2::runtime::AnyObject;
        use objc2_metal::{
            MTLClearColor, MTLLoadAction, MTLPrimitiveType, MTLRegion, MTLOrigin,
            MTLRenderPassDescriptor, MTLResourceOptions, MTLStorageMode, MTLStoreAction,
            MTLTextureDescriptor, MTLTextureUsage, MTLVertexDescriptor,
            MTLVertexFormat, MTLVertexStepFunction, MTLRenderPipelineDescriptor,
            MTLCommandBuffer as _, MTLTexture as _, MTLDevice as _,
            MTLRenderCommandEncoder as _, MTLRenderPipelineState as _,
        };
        use std::ptr::NonNull;

        // the generic decode: DYNAMIC on, everything else 0
        let mut key = [0u32; FRAGMENT_CONSTANTS];
        key[29] = 1;

        let prepared_fragment = prepare(RASTER_FRAG, Some(&key), false);
        let prepared_vertex = prepare(RASTER_VERT, None, false);

        // the bindings verbatim, keyed to the entry point's own name `main`
        let mut fragment_bindings = naga_msl::BindingMap::default();
        for binding in 0..3u32 {
            fragment_bindings.insert(
                naga::ResourceBinding { group: 0, binding },
                naga_msl::BindTarget { texture: Some(binding as u8), ..Default::default() },
            );
        }
        fragment_bindings.insert(
            naga::ResourceBinding { group: 0, binding: 3 },
            naga_msl::BindTarget { buffer: Some(3), ..Default::default() },
        );
        fragment_bindings.insert(
            naga::ResourceBinding { group: 0, binding: 4 },
            naga_msl::BindTarget { buffer: Some(4), ..Default::default() },
        );
        for slot in 10..13 {
            fragment_bindings.insert(
                naga::ResourceBinding { group: 0, binding: slot },
                naga_msl::BindTarget {
                    sampler: Some(naga_msl::BindSamplerTarget::Resource(slot as u8)),
                    ..Default::default()
                },
            );
        }
        let mut per_entry = naga_msl::EntryPointResourceMap::default();
        per_entry.insert(String::from("main"), naga_msl::EntryPointResources {
            resources: fragment_bindings,
            // the Tables' runtime-sized array needs its host-sized buffer
            sizes_buffer: Some(30),
            ..Default::default()
        });

        let (fragment_msl, fragment_entry) = glsl::compile_msl_stage_entry(
            &prepared_fragment, ShaderStage::Fragment, &[], Some(&per_entry),
        ).expect("the keyed fragment translates");
        let (vertex_msl, vertex_entry) = glsl::compile_msl_stage_entry(
            &prepared_vertex, ShaderStage::Vertex, &[], None,
        ).expect("the vertex translates");
        println!("=== the keyed raster.frag MSL ===\n{fragment_msl}\n=== end ===");

        let hardware = HardwareMetal::new().expect("the Metal raster constructs");
        let vertex_library = hardware.device
            .newLibraryWithSource_options_error(&NSString::from_str(&vertex_msl), None)
            .expect("the vertex MSL compiles");
        let fragment_library = hardware.device
            .newLibraryWithSource_options_error(&NSString::from_str(&fragment_msl), None)
            .expect("the fragment MSL compiles");
        let vertex_function = vertex_library.newFunctionWithName(&ns(&vertex_entry))
            .expect("the vertex main");
        let fragment_function = fragment_library.newFunctionWithName(&ns(&fragment_entry))
            .expect("the fragment main");

        // the pipeline: the descriptor A2's builder lays out, the real
        // functions and the float twin's depth (D32F_S8)
        let descriptor = MTLRenderPipelineDescriptor::new();
        let depth_format = depth_pixel_format(true);
        descriptor.setStencilAttachmentPixelFormat(depth_format);
        descriptor.setDepthAttachmentPixelFormat(depth_format);
        unsafe {
            let attachments = descriptor.colorAttachments();
            let colour: Retained<objc2_metal::MTLRenderPipelineColorAttachmentDescriptor> =
                attachments.objectAtIndexedSubscript(0);
            colour.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        }
        descriptor.setRasterSampleCount(1);
        descriptor.setVertexFunction(Some(vertex_function.as_ref()));
        descriptor.setFragmentFunction(Some(fragment_function.as_ref()));
        let vertex_descriptor: Retained<MTLVertexDescriptor> = MTLVertexDescriptor::vertexDescriptor();
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
            layout.setStride(96);
            layout.setStepFunction(MTLVertexStepFunction::PerVertex);
            descriptor.setVertexDescriptor(Some(&vertex_descriptor));
        }
        let pipeline = unsafe {
            hardware.device.newRenderPipelineStateWithDescriptor_error(&descriptor)
        }.expect("the keyed pipeline constructs");

        // the draw block: the six stages pass the vertex colour through
        let mut uniforms = [0u8; 1504];
        let mut word = |offset: usize, value: u32| {
            uniforms[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        };
        for stage in 0..6 {
            let at = stage * 16;
            word(at, 0x000F_000F);     // the sources: the all-ones pass form
            word(at + 4, 0x0000_0000); // the operands: none altered
            word(at + 8, 0x000F_000F); // the combiners: the pass shape
            word(at + 12, 0);
        }
        let uniforms_buffer = unsafe {
            hardware.device.newBufferWithBytes_length_options(
                NonNull::from(uniforms.as_ref()).cast(),
                uniforms.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }.expect("the uniforms allocate");
        let tables = vec![0u32; 65536];
        let tables_buffer = unsafe {
            hardware.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::from(tables.as_slice()).cast(),
                tables.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }.expect("the tables allocate");

        // six placed vertices: the left triangle green, the right red
        let vertex = |x: f32, y: f32, r: f32, g: f32, b: f32| -> [f32; 24] {
            let mut data = [0.0f32; 24];
            data[0] = x; data[1] = y; data[3] = 1.0;
            data[4] = r; data[5] = g; data[6] = b; data[7] = 1.0;
            data[14] = 0.5;
            data
        };
        let mut vertices = Vec::with_capacity(24 * 6);
        for (position, (r, g, b)) in [
            ((-1.0f32, -1.0f32), (0.0f32, 1.0f32, 0.0f32)),
            ((0.0f32, -1.0f32), (0.0f32, 1.0f32, 0.0f32)),
            ((-1.0f32, 1.0f32), (0.0f32, 1.0f32, 0.0f32)),
            ((0.0f32, -1.0f32), (1.0f32, 0.0f32, 0.0f32)),
            ((1.0f32, -1.0f32), (1.0f32, 0.0f32, 0.0f32)),
            ((1.0f32, 1.0f32), (1.0f32, 0.0f32, 0.0f32)),
        ] {
            vertices.extend(vertex(position.0, position.1, r, g, b));
        }
        let vertex_buffer = unsafe {
            hardware.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::from(vertices.as_slice()).cast(),
                vertices.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }.expect("the vertices allocate");

        // the targets: colour and depth (the pipeline declares both)
        const WIDTH: usize = 32;
        const HEIGHT: usize = 32;
        let color_descriptor = MTLTextureDescriptor::new();
        color_descriptor.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        unsafe {
            color_descriptor.setWidth(WIDTH);
            color_descriptor.setHeight(HEIGHT);
        }
        color_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        color_descriptor.setStorageMode(MTLStorageMode::Managed);
        let color_target = hardware.device.newTextureWithDescriptor(&color_descriptor)
            .expect("the colour target allocates");
        let depth_descriptor = MTLTextureDescriptor::new();
        depth_descriptor.setPixelFormat(MTLPixelFormat::Depth32Float_Stencil8);
        unsafe {
            depth_descriptor.setWidth(WIDTH);
            depth_descriptor.setHeight(HEIGHT);
        }
        depth_descriptor.setUsage(MTLTextureUsage::RenderTarget);
        depth_descriptor.setStorageMode(MTLStorageMode::Managed);
        let depth_target = hardware.device.newTextureWithDescriptor(&depth_descriptor)
            .expect("the depth target allocates");

        let pass = MTLRenderPassDescriptor::new();
        unsafe {
            let attachments = pass.colorAttachments();
            let colour: Retained<objc2_metal::MTLRenderPassColorAttachmentDescriptor> =
                attachments.objectAtIndexedSubscript(0);
            colour.setTexture(Some(color_target.as_ref()));
            colour.setLoadAction(MTLLoadAction::Clear);
            colour.setStoreAction(MTLStoreAction::Store);
            colour.setClearColor(MTLClearColor { red: 0.04, green: 0.05, blue: 0.09, alpha: 1.0 });
            let depth = pass.depthAttachment();
            depth.setTexture(Some(depth_target.as_ref()));
            depth.setLoadAction(MTLLoadAction::Clear);
            depth.setStoreAction(MTLStoreAction::Store);
            depth.setClearDepth(1.0);
        }
        let buffer = hardware.queue.commandBuffer().expect("the command buffer");
        let encoder = unsafe { buffer.renderCommandEncoderWithDescriptor(&pass) }
            .expect("the render encoder");
        unsafe {
            encoder.setRenderPipelineState(&pipeline);
            encoder.setVertexBuffer_offset_atIndex(Some(vertex_buffer.as_ref()), 0, 0);
            encoder.setFragmentBuffer_offset_atIndex(Some(uniforms_buffer.as_ref()), 0, 3);
            encoder.setFragmentBuffer_offset_atIndex(Some(tables_buffer.as_ref()), 0, 4);
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 6);
            let encoder_ref: &AnyObject = encoder.as_ref();
            let _: () = objc2::msg_send![encoder_ref, endEncoding];
        }
        buffer.commit();
        buffer.waitUntilCompleted();

        let mut pixels = [0u8; WIDTH * HEIGHT * 4];
        unsafe {
            color_target.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::from(pixels.as_mut_slice()).cast(),
                WIDTH * 4,
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize { width: WIDTH, height: HEIGHT, depth: 1 },
                },
                0,
            );
        }
        let screen = |x: usize, y: usize| {
            let at = (y * WIDTH + x) * 4;
            [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
        };
        assert_eq!(screen(8, 24), [0u8, 255, 0, 255], "the green left half");
        assert_eq!(screen(24, 24), [255u8, 0, 0, 255], "the red right half");
        assert_eq!(screen(8, 2), [10u8, 13, 23, 255], "the letterbox top");
        assert_eq!(screen(24, 2), [10u8, 13, 23, 255], "the letterbox top right");
    }

    /// A4's draw executor: the uniforms and the vertices staged through
    /// the ring's own staging, the pipeline built for the key, the draws
    /// run indexed — the whole app-side draw path on Metal, verified by
    /// the two halves' pixels.
    #[test]
    fn the_draw_executor_runs_placed_vertices() {
        use objc2_metal::{
            MTLRegion, MTLOrigin, MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage,
            MTLTexture as _,
        };
        use std::ptr::NonNull;

        let mut hardware = HardwareMetal::new().expect("the Metal raster constructs");

        const WIDTH: usize = 32;
        const HEIGHT: usize = 32;
        let color_descriptor = MTLTextureDescriptor::new();
        color_descriptor.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        unsafe {
            color_descriptor.setWidth(WIDTH);
            color_descriptor.setHeight(HEIGHT);
        }
        color_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        color_descriptor.setStorageMode(MTLStorageMode::Managed);
        let color_target = hardware.device.newTextureWithDescriptor(&color_descriptor)
            .expect("the colour target allocates");
        let depth_descriptor = MTLTextureDescriptor::new();
        depth_descriptor.setPixelFormat(MTLPixelFormat::Depth32Float_Stencil8);
        unsafe {
            depth_descriptor.setWidth(WIDTH);
            depth_descriptor.setHeight(HEIGHT);
        }
        depth_descriptor.setUsage(MTLTextureUsage::RenderTarget);
        depth_descriptor.setStorageMode(MTLStorageMode::Managed);
        let depth_target = hardware.device.newTextureWithDescriptor(&depth_descriptor)
            .expect("the depth target allocates");

        // the uniforms: the six pass-through stages, the words exactly as
        // the draw's registers carry them
        let mut registers = vec![0u32; 0x300];
        for base in [0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            registers[base] = 0x000F_000F;
        }
        let mut uniform_words = Vec::new();
        draw_uniform_block(&mut uniform_words, &registers, 0, 0,
            super::super::DepthMap { scale: 1.0, offset: 0.0, w_buffer: false }, None);
        // the placed vertices, packed the Vulkan raster's own way
        let vertex = |x: f32, y: f32, r: f32, g: f32, b: f32| -> [f32; 24] {
            let mut data = [0.0f32; 24];
            data[0] = x; data[1] = y; data[3] = 1.0;
            data[4] = r; data[5] = g; data[6] = b; data[7] = 1.0;
            data[14] = 0.5;
            data
        };
        let mut vertices = Vec::with_capacity(24 * 6);
        for (position, (r, g, b)) in [
            ((-1.0f32, -1.0f32), (0.0f32, 1.0f32, 0.0f32)),
            ((0.0f32, -1.0f32), (0.0f32, 1.0f32, 0.0f32)),
            ((-1.0f32, 1.0f32), (0.0f32, 1.0f32, 0.0f32)),
            ((0.0f32, -1.0f32), (1.0f32, 0.0f32, 0.0f32)),
            ((1.0f32, -1.0f32), (1.0f32, 0.0f32, 0.0f32)),
            ((1.0f32, 1.0f32), (1.0f32, 0.0f32, 0.0f32)),
        ] {
            vertices.extend(vertex(position.0, position.1, r, g, b));
        }
        let indices: [u32; 6] = [0, 1, 2, 3, 4, 5];

        let mut draw = PlacedDraw {
            vertices: &vertices,
            indices: &indices[..3],
            textures: [None; 3],
            target: color_target.as_ref(),
            clear: true,
            depth: Some(depth_target.as_ref()),
            width: WIDTH as u32,
            height: HEIGHT as u32,
        };
        let key = PipelineKey { mask: 0xF, depth: true, float_depth: true, vertex: VertexStage::Placed };
        hardware.draw_placed(&draw, &key, &uniform_words, &[]).expect("the first draw runs");
        draw.clear = false;
        draw.indices = &indices[3..];
        let red_only = PipelineKey { mask: 1, ..key };
        hardware.draw_placed(&draw, &red_only, &uniform_words, &[]).expect("the second draw preserves unwritable channels");

        // A no-depth pass with all channels masked must preserve the image.
        draw.depth = None;
        let no_writes = PipelineKey { mask: 0, depth: false, ..key };
        hardware.draw_placed(&draw, &no_writes, &uniform_words, &[]).expect("color-only pipeline");
        assert!(hardware.draw_placed(&draw, &key, &uniform_words, &[]).is_err());
        let unsupported = PipelineKey { vertex: VertexStage::Interpreted, ..no_writes };
        assert!(hardware.draw_placed(&draw, &unsupported, &uniform_words, &[]).is_err());

        let mut pixels = [0u8; WIDTH * HEIGHT * 4];
        unsafe {
            color_target.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::from(pixels.as_mut_slice()).cast(),
                WIDTH * 4,
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize { width: WIDTH, height: HEIGHT, depth: 1 },
                },
                0,
            );
        }
        let screen = |x: usize, y: usize| {
            let at = (y * WIDTH + x) * 4;
            [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
        };
        assert_eq!(screen(8, 24), [0u8, 255, 0, 255], "the green left half");
        assert_eq!(screen(24, 24), [255u8, 13, 23, 255], "only red changes; green, blue and alpha stay loaded");
        assert_eq!(screen(8, 2), [10u8, 13, 23, 255], "the letterbox top");
        assert_eq!(screen(24, 2), [10u8, 13, 23, 255], "the letterbox top right");

        // Exercise all three texture/sampler slots through the real fragment shader.
        let sampler_desc = objc2_metal::MTLSamplerDescriptor::new();
        sampler_desc.setMinFilter(objc2_metal::MTLSamplerMinMagFilter::Nearest);
        sampler_desc.setMagFilter(objc2_metal::MTLSamplerMinMagFilter::Nearest);
        let sampler = hardware.device.newSamplerStateWithDescriptor(&sampler_desc).unwrap();
        let tex_desc = MTLTextureDescriptor::new();
        tex_desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        tex_desc.setStorageMode(MTLStorageMode::Managed);
        tex_desc.setUsage(MTLTextureUsage::ShaderRead);
        unsafe { tex_desc.setWidth(1); tex_desc.setHeight(1); }
        let colors = [[255u8, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        let textures: Vec<_> = colors.iter().map(|pixel| {
            let texture = hardware.device.newTextureWithDescriptor(&tex_desc).unwrap();
            unsafe {
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: objc2_metal::MTLSize { width: 1, height: 1, depth: 1 } },
                    0, NonNull::from(&pixel[..]).cast(), 4);
            }
            texture
        }).collect();
        draw.textures = std::array::from_fn(|unit| Some(TextureBinding {
            texture: textures[unit].as_ref(), sampler: sampler.as_ref(),
        }));
        let mut sampled_words = [0u32; 408];
        for stage in 0..6 { sampled_words[stage * 8] = 0x000F_000F; }
        for unit in 0..3 {
            let source = 3 + unit as u32;
            sampled_words[0] = source | (source << 16);
            sampled_words[51] = 1 << unit;
            let sampling_key = PipelineKey { mask: 15, ..no_writes };
            hardware.draw_placed(&draw, &sampling_key, &sampled_words, &[0; 2]).unwrap();
            let mut pixel = [0u8; 4];
            unsafe {
                color_target.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    NonNull::from(&mut pixel[..]).cast(), 4,
                    MTLRegion { origin: MTLOrigin { x: 24, y: 24, z: 0 },
                        size: objc2_metal::MTLSize { width: 1, height: 1, depth: 1 } }, 0);
            }
            assert_eq!(pixel, colors[unit], "texture/sampler unit {unit}");
        }
        // Fog reads a runtime-sized table and the last vec4 of Draw.
        // With a 0.5 factor, blue mixed with yellow produces grey.
        sampled_words[48] = 5;
        sampled_words[404] = 0x0000_FFFF;
        let mut tables = vec![0u32; (7552 + 128) * 2];
        for pair in tables.chunks_exact_mut(2) { pair[0] = 0.5f32.to_bits(); }
        hardware.draw_placed(&draw, &PipelineKey { mask: 15, ..no_writes }, &sampled_words, &tables).unwrap();
        let mut fog_pixel = [0u8; 4];
        unsafe {
            color_target.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::from(&mut fog_pixel[..]).cast(), 4,
                MTLRegion { origin: MTLOrigin { x: 24, y: 24, z: 0 },
                    size: objc2_metal::MTLSize { width: 1, height: 1, depth: 1 } }, 0);
        }
        assert_eq!(fog_pixel, [127, 127, 127, 255], "Draw tail and Tables size binding");
        draw.textures[2] = None;
        assert!(hardware.draw_placed(&draw, &no_writes, &sampled_words, &[]).is_err());
        assert!(hardware.draw_placed(&draw, &no_writes, &[], &[]).is_err());
    }

    #[test]
    fn the_display_transfer_executor_runs_the_guest_ramp() {
        use objc2_metal::MTLTexture as _;

        let mut hardware = HardwareMetal::new().expect("the Metal raster constructs");
        let mut memory = HostMemory::new();

        // the input: 16x16 grey ramp (the pixel (x, y) carries x)
        let input_addr = 0x0100_0000u32;
        let mut bytes = vec![0u8; (16 * 16 * 4) as usize];
        for (index, pixel) in bytes.chunks_exact_mut(4).enumerate() {
            let x = (index % 16) as u8;
            pixel.copy_from_slice(&[255u8, x, x, x]);
        }
        memory.write_raw(input_addr as u64, &bytes);

        let transfer = crate::raster::Transfer {
            input: input_addr, output: input_addr + 16 * 16 * 4,
            input_width: 16, output_width: 8, output_height: 8,
            copy: (8, 8), scale: (2, 2),
            flip: false, input_linear: true, output_tiled: false,
            input_format: ColorFormat::Rgba8, output_format: ColorFormat::Rgba8,
        };
        hardware.display_transfer(&mut memory, &transfer).expect("the transfer runs");

        // two surfaces over the run: the input's at index 0 with its shadow,
        // the output's at 1 (the transfer's own writes carried the texture)
        assert_eq!(hardware.surfaces.len(), 2, "the in and the out");

        // the output's pixels read back the honest averaged shape
        let output = &hardware.surfaces[1];
        let mut pixels = [0u8; 8 * 8 * 4];
        unsafe {
            output.texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::from(pixels.as_mut_slice()).cast(),
                32,
                objc2_metal::MTLRegion {
                    origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize { width: 8 as usize, height: 8 as usize, depth: 1 },
                },
                0,
            );
        }
        let screen = |x: usize, y: usize| {
            let at = (y * 8 + x) * 4;
            [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
        };
        // the debug: the whole read-back's red bytes ROW-MAJOR
        for y in 0..8u8 {
            println!("row {y}: {:?}", (0..8usize).map(|x| screen(x as usize, y as usize)[0]).collect::<Vec<_>>());
        }
        assert_eq!(screen(0, 0)[0], 1, "the average of columns 0 and 1");
        assert_eq!(screen(7, 0)[0], 15, "the average of the ramp's last columns");
        assert_eq!(screen(0, 7)[0], 1, "the bottom rows' own average");
        assert_eq!(screen(4, 4)[0], 9, "the average of 8 and 9 past (34+2)/4");
        assert_eq!(pixels[3], 255, "the alpha stays full");
    }

    #[test]
    fn display_transfer_compute_runs_and_copies_pixels() {
        // A3c: the compute shaders' binding proof — the display transfer's
        // own MSL compiles with the resource map and the pipeline state
        // (the draw's pixel assertions live in the full path below).
        let rack = rack().expect("the rack translates");
        let transfer = &rack.transfer_comp;
        assert!(transfer.msl.contains("[[texture(0)]]"), "the source sits at texture 0");
        assert!(transfer.msl.contains("[[texture(1)]]"), "the target sits at texture 1");
        assert!(transfer.msl.contains("[[buffer(24)]]") || transfer.msl.contains("constant Transfer& transfer [[buffer(24)]]"), "the push constants ride 24");
        assert!(transfer.entry.starts_with("main"), "the entry resolves");

        let upright_image = &rack.upright_image_comp;
        // the moved SSBO and the push constants ride their slots
        assert!(upright_image.msl.contains("constant Screen& screen [[buffer(24)]]"), "the upright's push at 24");
        assert!(upright_image.msl.contains("[[buffer(24)]]"), "the push constants ride 24");
        let hardware = HardwareMetal::new().expect("the Metal raster constructs");
        for (shader, compute) in [
            ("transfer.comp", true), ("upright_image.comp", true),
        ] {
            let library = hardware.device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(&rack.entries().iter()
                        .find(|translated| translated.name == shader).unwrap().msl),
                    None,
                ).expect(shader);
            let kernel = library.newFunctionWithName(
                &ns(&rack.entries().iter()
                    .find(|translated| translated.name == shader).unwrap().entry)
            ).expect(shader);
            if compute {
                let compute_state = unsafe {
                    hardware.device.newComputePipelineStateWithFunction_error(kernel.as_ref())
                }.expect(shader);
                use objc2_metal::MTLComputePipelineState as _;
                assert!(compute_state.threadExecutionWidth() >= 1, "the dispatch can run");
            }
        }
    }
}

/// the test shaders: a triangle drawn plain, so the pixels prove the
/// pipeline runs the raster's way (NDC positions, a flat colour out).
fn tri_vertex() -> String {
    String::from(r#"
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 out_color;
void main() {
    out_color = color;
    gl_Position = vec4(position, 1.0);
}
"#
    )
}

fn tri_fragment() -> String {
    String::from(r#"
#version 450
layout(location = 0) in vec4 in_color;
layout(location = 0) out vec4 frag_color;
void main() {
    frag_color = in_color;
}
"#
    )
}
