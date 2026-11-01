//! The Metal presenter, phase M1.0 (present parity): the game's screens —
//! the CPU-drawn framebuffers — become Metal textures and are drawn onto the
//! window's CAMetalLayer through the shared `place()` layout, the same way
//! the Vulkan presenter does. The raster itself stays on the software path
//! (the Vulkan GPU raster needs its own M1.1): this phase proves the
//! complete presentation chain with real frames.
//!
//! # Device lifecycle
//! The device, the command queue and the layer are created once with the
//! window's NSView; textures are recreated when a screen's size changes, and
//! the vertex buffer when the window resizes. Frames upload the screens with
//! `replaceRegion` and draw the quads, the overlay unrendered (logged once).
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLStoreAction, MTLTextureDescriptor,
    MTLCreateSystemDefaultDevice, MTLDevice as _, MTLLibrary as _,
    MTLCommandQueue as _, MTLCommandBuffer as _, MTLRenderCommandEncoder as _,
};
use objc2_quartz_core::{CAMetalDrawable as _, CAMetalLayer};
use super::{place, ScreenImage, ScreenLayout, Stereo};
use super::ScreenFilter;

use crate::backend::Shown;

const SHADER: &str = r"#include <metal_stdlib>
using namespace metal;
struct ScreenRect { float4 rect; }; // x0, y0, x1, y1 in clip space
struct SVIn {
    float4 position [[position]];
    float2 uv [[user(locn0)]];
};
vertex SVIn screen_vertex(const device ScreenRect& quad [[buffer(0)]], uint vid [[vertex_id]]) {
    // the six corners walk two triangles over the rect: (LT RT BR)(BR LB LT);
    // the texture's first row is the screen's top, so v = 0 at the top edge
    SVIn out;
    switch (vid) {
        case 0: out.position = float4(quad.rect.x, quad.rect.w, 0.0, 1.0); out.uv = float2(0.0, 0.0); break;
        case 1: out.position = float4(quad.rect.z, quad.rect.w, 0.0, 1.0); out.uv = float2(1.0, 0.0); break;
        case 2: out.position = float4(quad.rect.z, quad.rect.y, 0.0, 1.0); out.uv = float2(1.0, 1.0); break;
        case 3: out.position = float4(quad.rect.z, quad.rect.y, 0.0, 1.0); out.uv = float2(1.0, 1.0); break;
        case 4: out.position = float4(quad.rect.x, quad.rect.y, 0.0, 1.0); out.uv = float2(0.0, 1.0); break;
        default: out.position = float4(quad.rect.x, quad.rect.w, 0.0, 1.0); out.uv = float2(0.0, 0.0); break;
    }
    return out;
}
fragment half4 screen_fragment(texture2d<float> tex [[texture(0)]], sampler samp [[sampler(0)]], SVIn in [[stage_in]]) {
    return half4(tex.sample(samp, in.uv));
}
";

// a rotated screen texture on a quad: position in drawables' pixels,
// uv in texture space, six per quad.
const QUADS_PER_FRAME: usize = 4;

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

struct ScreenTexture {
    texture: Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>,
    width: u32,
    height: u32,
}

pub struct MetalPresenter {
    device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>>,
    pub layer: Retained<CAMetalLayer>,
    sampler: Retained<ProtocolObject<dyn objc2_metal::MTLSamplerState>>,
    // the screen textures, top / right / bottom
    textures: [Option<ScreenTexture>; 3],
    // and the frame's vertex buffer, rebuilt on resize
    quads: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    quads_for: (u32, u32),
    scale: f64,
    width: u32,
    height: u32,
    arrangement: ScreenLayout,
    top_width: f32,
    integer: bool,
    stereo: Stereo,
    frames: u64,
    overlay_warned: bool,
    started: std::time::Instant,
}

fn index_of(shown: Shown) -> usize {
    match shown {
        Shown::Top => 0,
        Shown::Right => 1,
        Shown::Bottom => 2,
        Shown::Both => 0,
    }
}

impl MetalPresenter {
    pub fn new(ns_view: *mut core::ffi::c_void, width: u32, height: u32, scale: f64) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device on this Mac")?;
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;

        let source = ns(SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|error| format!("no Metal library: {error}"))?;
        let vertex = library
            .newFunctionWithName(&ns("screen_vertex"))
            .ok_or("no screen_vertex function")?;
        let fragment = library
            .newFunctionWithName(&ns("screen_fragment"))
            .ok_or("no screen_fragment function")?;
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        let _attachments = unsafe { descriptor.colorAttachments() };
        let _attachment0: Retained<objc2_metal::MTLRenderPipelineColorAttachmentDescriptor> = unsafe { _attachments.objectAtIndexedSubscript(0) };
        _attachment0.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        let pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| format!("no pipeline state: {error}"))?;

        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&*device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setMaximumDrawableCount(3);
        // stays framebufferOnly: CoreAnimation keeps its zero-copy present,
        // and the pipeline samples OUR Managed textures, never the drawable
        layer.setFramebufferOnly(true);
        layer.setDrawableSize(CGSize::new(
            (width.max(1) * scale as u32) as f64,
            (height.max(1) * scale as u32) as f64,
        ));
        unsafe {
            let view: &AnyObject = &*(ns_view as *const AnyObject);
            let _: () = objc2::msg_send![view, setLayer: &*layer];
            let _: () = objc2::msg_send![view, setWantsLayer: true];
        }

        let sampler_descriptor = objc2_metal::MTLSamplerDescriptor::new();
        sampler_descriptor.setMinFilter(objc2_metal::MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.setMagFilter(objc2_metal::MTLSamplerMinMagFilter::Linear);
        let sampler = device
            .newSamplerStateWithDescriptor(&sampler_descriptor)
            .ok_or("no Metal sampler")?;

        Ok(MetalPresenter {
            device,
            queue,
            pipeline,
            layer,
            sampler,
            textures: [None, None, None],
            quads: None,
            quads_for: (0, 0),
            scale,
            width,
            height,
            arrangement: ScreenLayout::default(),
            top_width: 1.0,
            integer: false,
            stereo: Stereo::default(),
            frames: 0,
            overlay_warned: false,
            started: std::time::Instant::now(),
        })
    }

    pub fn name(&self) -> &'static str {
        "metal"
    }

    pub fn resize(&mut self, width: u32, height: u32, scale: f64) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.scale = scale;
        self.layer.setDrawableSize(CGSize::new(
            (self.width * self.scale as u32) as f64,
            (self.height * self.scale as u32) as f64,
        ));
        self.quads = None;
    }

    pub fn set_scaling(&mut self, _filter: ScreenFilter, _integer: bool) {
        self.integer = _integer;
    }

    pub fn set_layout(&mut self, arrangement: ScreenLayout) {
        self.arrangement = arrangement;
    }

    pub fn set_top_width(&mut self, width: f32) {
        self.top_width = width;
    }

    pub fn set_stereo(&mut self, stereo: Stereo) {
        self.stereo = stereo;
    }

    fn texture_for(&mut self, index: usize, width: u32, height: u32) -> Option<&Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>> {
        let sized = |texture: &ScreenTexture| texture.width == width && texture.height == height;
        if !self.textures[index].as_ref().is_some_and(sized) {
            let descriptor = MTLTextureDescriptor::new();
            descriptor.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
            unsafe {
                descriptor.setWidth((width.max(1)) as usize);
                descriptor.setHeight((height.max(1)) as usize);
                // the AGX driver's compressed-contents path crashes writing a
                // fresh Private texture on G13 (writeRegion faults at 0): the
                // screens come and go at every frame, so linear storage both
                // avoids the fault and uploads faster
                descriptor.setAllowGPUOptimizedContents(false);
            }
            descriptor.setStorageMode(objc2_metal::MTLStorageMode::Managed);
            descriptor.setUsage(objc2_metal::MTLTextureUsage::ShaderRead);
            let texture = self
                .device
                .newTextureWithDescriptor(&descriptor)
                .ok_or("texture creation failed")
                .ok()?;
            self.textures[index] = Some(ScreenTexture { texture, width, height });
        }
        self.textures[index].as_ref().map(|screen_texture| &screen_texture.texture)
    }

    fn upload_texture(&mut self, index: usize, bytes: &[u8], width: u32, height: u32) -> bool {
        let Some(texture) = self.texture_for(index, width, height) else { return false };
        let bytes_per_row = width * 4;
        use objc2_metal::MTLTexture as _;
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                objc2_metal::MTLRegion {
                    origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize {
                        width: width as usize,
                        height: height as usize,
                        depth: 1,
                    },
                },
                0,
                std::ptr::NonNull::from(bytes).cast(),
                bytes_per_row as usize,
            );
        }
        true
    }

    /// the frame: the screens in layers, textures uploaded, quads drawn.
    pub fn present_screens(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
        right: ScreenImage<'_>,
        stereo: Stereo,
    ) -> Result<(), String> {
        if self.stereo != stereo {
            self.stereo = stereo;
        }
        let quads_needed = unsafe { self.layer.drawableSize() };
        if (quads_needed.width as u32, quads_needed.height as u32) != self.quads_for && !self.overlay_warned {
            // rebuild the quad buffer beside the layer's own size
            self.quads_for = (quads_needed.width as u32, quads_needed.height as u32);
            self.quads.take();
        }

        let places = place(
            self.quads_for.0, self.quads_for.1, self.arrangement, self.top_width,
            self.integer, self.stereo, false,
        );

        // the screen textures the quads sample, uploaded from the CPU frames
        let mut textures: [Option<Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>>; 3] = [None, None, None];
        for (_, shown) in &places {
            let image = match shown {
                Shown::Top | Shown::Both => &top,
                Shown::Right => if right.is_empty() { &top } else { &right },
                Shown::Bottom => &bottom,
            };
            let index = index_of(*shown);
            if image.is_empty() || image.gpu.is_some() {
                continue;
            }
            if self.upload_texture(index, image.pixels, image.width, image.height) {
                if let Some(screen) = &self.textures[index] {
                    textures[index] = Some(screen.texture.clone());
                }
            }
        }

        let Some(drawable) = self.layer.nextDrawable() else {
            return Err(String::from("no drawable this frame"));
        };
        let pass = MTLRenderPassDescriptor::new();
        let attachments = unsafe { pass.colorAttachments() };
        let attachment: Retained<objc2_metal::MTLRenderPassColorAttachmentDescriptor> =
            unsafe { attachments.objectAtIndexedSubscript(0) };
        attachment.setTexture(Some(drawable.texture().as_ref()));
        attachment.setLoadAction(MTLLoadAction::Clear);
        attachment.setStoreAction(MTLStoreAction::Store);
        attachment.setClearColor(MTLClearColor {
            red: 0.04,
            green: 0.05,
            blue: 0.09,
            alpha: 1.0,
        });

        let Some(buffer) = self.queue.commandBuffer() else {
            return Err(String::from("no command buffer"));
        };
        let Some(encoder) = (unsafe { buffer.renderCommandEncoderWithDescriptor(&pass) }) else {
            return Err(String::from("no encoder"));
        };
        unsafe {
            encoder.setRenderPipelineState(&self.pipeline);

            // one draw per place: the vertex shader recognizes the quad's
            // corners out of a 16-byte rect, no data buffer to carry, no
            // pointer fetch over the old vertex array
            encoder.setFragmentSamplerState_atIndex(Some(self.sampler.as_ref()), 0);
            let (cols, rows) = (self.quads_for.0 as f32, self.quads_for.1 as f32);
            for (viewport, shown) in &places {
                let image = match shown {
                    Shown::Top | Shown::Both => &top,
                    Shown::Right => if right.is_empty() { &top } else { &right },
                    Shown::Bottom => &bottom,
                };
                if image.is_empty() || image.gpu.is_some() {
                    continue;
                }
                let index = index_of(*shown);
                if textures[index].is_none() {
                    continue;
                }
                encoder.setFragmentTexture_atIndex(textures[index].as_deref(), 0);
                let rect = [
                    viewport.x / cols * 2.0 - 1.0,
                    1.0 - (viewport.y + viewport.height) / rows * 2.0,
                    (viewport.x + viewport.width) / cols * 2.0 - 1.0,
                    1.0 - viewport.y / rows * 2.0,
                ];
                encoder.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(rect.as_slice()).cast(),
                    rect.len() * 4,
                    0,
                );
                encoder.drawPrimitives_vertexStart_vertexCount(
                    objc2_metal::MTLPrimitiveType::Triangle,
                    0,
                    6,
                );
            }
            let encoder_ref: &AnyObject = encoder.as_ref();
            let _: () = objc2::msg_send![encoder_ref, endEncoding];
            // the present itself: without this call the compositor never sees
            // the frame — the window stays white while the game runs behind
            let buffer_ref: &AnyObject = buffer.as_ref();
            let drawable_object: &AnyObject = drawable.as_ref();
            let _: () = objc2::msg_send![buffer_ref, presentDrawable: drawable_object];
        }
        buffer.commit();
        self.frames += 1;
        if self.frames == 1 {
            log::info!("metal present v1: first frame dipped through, {} places", places.len());
        }
        Ok(())
    }

    pub fn report(&self) -> String {
        let seconds = self.started.elapsed().as_secs_f64();
        let fps = if self.frames > 1 { (self.frames - 1) as f64 / seconds } else { 0.0 };
        format!("metal present: {} frames in {:.3}s ({:.1} fps)", self.frames, seconds, fps)
    }
}
