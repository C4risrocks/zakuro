//! The Metal presenter, phase M1.0 (present parity) with M1.0.5 (the
//! decoupled present): the game's screens — the CPU-drawn framebuffers —
//! become Metal textures and are drawn onto the window's CAMetalLayer
//! through the shared `place()` layout, the same way the Vulkan presenter
//! does. The raster itself stays on the software path (the Vulkan GPU
//! raster needs its own M1.1): this phase proves the complete presentation
//! chain with real frames.
//!
//! # The present runs on its own thread
//! Measuring the real game showed the presenter eating the emulation's
//! budget: 85% of the frame loop slept inside `nextDrawable`, waiting for
//! the display's vsync, right on the thread the software raster runs on —
//! MM3D dropped from 54 fps to 26. The whole Metal bundle (device, queue,
//! layer, pipeline, textures) lives on a dedicated worker thread now; the
//! app's thread only verifies, screenshots and sends the frame's pixels
//! through a channel (one frame in flight, so the wait and the raster
//! overlap instead of queueing). The layout's come with each frame.
//!
//! # Device lifecycle
//! The device, the command queue and the layer are created once with the
//! window's NSView on the main thread and moved to the worker; textures
//! are recreated when a screen's size changes, and the vertex data goes
//! inline with the draw (a 16-byte rect per screen).        The overlay
//! stays unrendered (egui on Metal is M2.1).
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLStoreAction, MTLTextureDescriptor,
    MTLCreateSystemDefaultDevice, MTLDevice as _, MTLLibrary as _,
    MTLCommandQueue as _, MTLCommandBuffer as _, MTLRenderCommandEncoder as _,
    MTLCommandEncoder as _, MTLBlitCommandEncoder as _,
};
use objc2_quartz_core::{CAMetalDrawable as _, CAMetalLayer};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use super::{place, ScreenImage, ScreenLayout, Stereo};

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
        default: out.position = float4(quad.rect.x, quad.rect.w, 0.0, 1.0); out.uv = float2(0.0, 0.0); break;    }
    return out;
}
fragment half4 screen_fragment(texture2d<float> tex [[texture(0)]], sampler samp [[sampler(0)]], SVIn in [[stage_in]]) {
    return half4(tex.sample(samp, in.uv));
}
";

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

struct ScreenTexture {
    texture: Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>,
    width: u32,
    height: u32,
}

/// the layout settings the worker consults per frame, mutated from the
/// main thread's setters and undoing the whole point if assembled off it
#[derive(Clone, Copy)]
struct Config {
    arrangement: ScreenLayout,
    top_width: f32,
    integer: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { arrangement: ScreenLayout::default(), top_width: 1.0, integer: false }
    }
}

/// one presentation in flight: the frame's pixels and how the 3D shows
struct Frame {
    top: Vec<u8>,
    top_width: u32,
    top_height: u32,
    bottom: Vec<u8>,
    bottom_width: u32,
    bottom_height: u32,
    right: Vec<u8>,
    right_width: u32,
    right_height: u32,
    stereo: Stereo,
    width: u32,
    height: u32,
    scale: f64,
}

unsafe impl Send for Frame {}

/// CAMetalLayer wrapped for the thread move: Metal's calls on the layer
/// (nextDrawable, drawableSize) are documented thread-traversing, the way
/// wgpu and the wider Metal tooling acquire and present off the main
/// thread; the AppKit bits (installing on the view) stay on the main one
struct SendLayer(Retained<CAMetalLayer>);
unsafe impl Send for SendLayer {}

/// the whole Metal system, owned by the worker thread
struct Metal {
    layer: SendLayer,
    device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>>,
    sampler: Retained<ProtocolObject<dyn objc2_metal::MTLSamplerState>>,
    // two texture sets ping-pong so the upload of a frame never points a
    // write at a texture the previous job's draw is still sampling
    textures: [[Option<ScreenTexture>; 3]; 2],
    flip: bool,
    // the drawable size the layer already carries, so it only gets set
    // when the window actually changed it
    resized: (f64, f64),
}

impl Metal {
    fn texture_for(
        &mut self,
        index: usize,
        width: u32,
        height: u32,
    ) -> Option<&Retained<ProtocolObject<dyn objc2_metal::MTLTexture>>> {
        let sized = |texture: &ScreenTexture| {
            texture.width == width && texture.height == height
        };
        let set = usize::from(self.flip);
        if !self.textures[set][index].as_ref().is_some_and(sized) {
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
            self.textures[set][index] = Some(ScreenTexture { texture, width, height });
        }
        self.textures[set][index]
            .as_ref()
            .map(|screen_texture| &screen_texture.texture)
    }

    fn upload(
        &mut self,
        index: usize,
        bytes: &[u8],
        width: u32,
        height: u32,
    ) -> bool {
        let Some(texture) = self.texture_for(index, width, height) else { return false };
        let bytes_per_row = width * 4;
        use objc2_metal::MTLTexture as _;
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                objc2_metal::MTLRegion {
                    origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: objc2_metal::MTLSize { width: width as usize, height: height as usize, depth: 1 },
                },
                0,
                std::ptr::NonNull::from(bytes).cast(),
                bytes_per_row as usize,
            );
        }
        true
    }
}

// the bundle moves to the worker exactly once and all of its Metal objects
// are then used from that one thread — the discipline the channel gives
unsafe impl Send for Metal {}

/// the worker's loop: one frame at a time until the channel closes
fn present_worker(mut metal: Metal, jobs: mpsc::Receiver<Frame>, config: Arc<Mutex<Config>>, done: Arc<AtomicU64>) {
    while let Ok(frame) = jobs.recv() {
        // the window's size changed? apply it before the layout reads — the
        // setter runs only on real changes: calling it every frame would
        // make CoreAnimation re-prepare the drawables on the main thread
        let stamp = (
            (frame.width * frame.scale as u32) as f64,
            (frame.height * frame.scale as u32) as f64,
        );
        if (metal.resized.0 - stamp.0).abs() > f64::EPSILON
            || (metal.resized.1 - stamp.1).abs() > f64::EPSILON
        {
            metal.layer.0.setDrawableSize(CGSize::new(stamp.0, stamp.1));
            metal.resized = stamp;
        }
        let (width, height) = {
            let size = unsafe { metal.layer.0.drawableSize() };
            (size.width as u32, size.height as u32)
        };
        let cfg = *config.lock().unwrap();
        let places = place(width, height, cfg.arrangement, cfg.top_width, cfg.integer, frame.stereo, false);
        metal.flip = !metal.flip;

        let drawable = unsafe { metal.layer.0.nextDrawable() };
        let Some(drawable) = drawable else {
            continue;
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

        let Some(buffer) = metal.queue.commandBuffer() else {
            continue;
        };
        let Some(encoder) = (unsafe { buffer.renderCommandEncoderWithDescriptor(&pass) }) else {
            continue;
        };
        unsafe {
            encoder.setRenderPipelineState(&metal.pipeline);
            encoder.setFragmentSamplerState_atIndex(Some(metal.sampler.as_ref()), 0);

            // one draw per place: the vertex shader recognizes the quad's
            // corners out of a 16-byte rect, no data buffer to carry
            let (cols, rows) = (width as f32, height as f32);
            for (viewport, shown) in &places {
                let image = match shown {
                    Shown::Top | Shown::Both => &frame.top,
                    Shown::Right => if frame.right.is_empty() { &frame.top } else { &frame.right },
                    Shown::Bottom => &frame.bottom,
                };
                if image.is_empty() {
                    continue;
                }
                let (w, h) = match shown {
                    Shown::Top | Shown::Both => (frame.top_width, frame.top_height),
                    Shown::Right => if frame.right.is_empty() { (frame.top_width, frame.top_height) } else { (frame.right_width, frame.right_height) },
                    Shown::Bottom => (frame.bottom_width, frame.bottom_height),
                };
                let index = index_of(*shown);
                if !metal.upload(index, image, w, h) {
                    continue;
                }
                let Some(texture) = metal.texture_for(index, w, h) else { continue; };
                encoder.setFragmentTexture_atIndex(Some(texture.as_ref()), 0);
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
        done.fetch_add(1, Ordering::Relaxed);
        let _ = frame;
    }
}

fn index_of(shown: Shown) -> usize {
    match shown {
        Shown::Top => 0,
        Shown::Right => 1,
        Shown::Bottom => 2,
        Shown::Both => 0,
    }
}

/// the presenter half the app talks to: everything heavy lives on the
/// worker thread; sending a frame never blocks on the display
pub struct MetalPresenter {
    jobs: Option<mpsc::SyncSender<Frame>>,
    config: Arc<Mutex<Config>>,
    done: Arc<AtomicU64>,
    // the window's geometry, updated by resize() and stamped on each job
    // (the worker re-applies the layer's drawable size with it)
    width: u32,
    height: u32,
    scale: f64,
    started: std::time::Instant,
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

        let metal = Metal {
            layer: SendLayer(layer),
            device,
            queue,
            pipeline,
            sampler,
            textures: [[None, None, None], [None, None, None]],
            flip: false,
            // seeded with the initial size so the worker's first frame
            // doesn't re-set what new() already set
            resized: (
                (width.max(1) * scale as u32) as f64,
                (height.max(1) * scale as u32) as f64,
            ),
        };

        let config = Arc::new(Mutex::new(Config::default()));
        let done = Arc::new(AtomicU64::new(0));
        // one frame in flight: the worker takes the next as soon as it has
        // committed the current one, while the app's thread keeps rastering
        let (jobs_tx, jobs_rx) = mpsc::sync_channel(1);
        let spawn_config = config.clone();
        let spawn_done = done.clone();
        let worker = std::thread::Builder::new()
            .name("metal-present".to_owned())
            .spawn(move || present_worker(metal, jobs_rx, spawn_config, spawn_done))
            .map_err(|error| format!("no Metal present thread: {error}"))?;

        Ok(MetalPresenter {
            jobs: Some(jobs_tx),
            config,
            done,
            width: width.max(1),
            height: height.max(1),
            scale,
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
        // the worker re-applies the layer's size with the next job, so the
        // resize trail is at most a frame behind the window
    }

    pub fn set_scaling(&mut self, _filter: crate::backend::ScreenFilter, integer: bool) {
        self.config.lock().unwrap().integer = integer;
    }

    pub fn set_layout(&mut self, arrangement: ScreenLayout) {
        self.config.lock().unwrap().arrangement = arrangement;
    }

    pub fn set_top_width(&mut self, width: f32) {
        self.config.lock().unwrap().top_width = width;
    }

    pub fn set_stereo(&mut self, stereo: Stereo) {
        let _ = stereo; // the frame's own stereo rides with each job
    }

    /// the frame: verify, hand the pixels to the worker, keep the ration
    pub fn present_screens(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
        right: ScreenImage<'_>,
        stereo: Stereo,
    ) -> Result<(), String> {
        let hand = |image: &ScreenImage<'_>| {
            // GPU-borne images can't be shown by this path (M1.1's job);
            // they travel empty so the place's quad just doesn't draw
            if image.gpu.is_some() { Vec::new() } else { image.pixels.to_vec() }
        };
        let frame = Frame {
            top: hand(&top),
            top_width: top.width,
            top_height: top.height,
            bottom: hand(&bottom),
            bottom_width: bottom.width,
            bottom_height: bottom.height,
            right: hand(&right),
            right_width: right.width,
            right_height: right.height,
            stereo,
            width: self.width,
            height: self.height,
            scale: self.scale,
        };
        let jobs = self.jobs.as_ref().ok_or("no Metal present thread")?;
        if jobs.send(frame).is_err() {
            return Err("the Metal present thread stopped taking frames".to_owned());
        }
        Ok(())
    }

    pub fn report(&self) -> String {
        let seconds = self.started.elapsed().as_secs_f64();
        let frames = self.done.load(Ordering::Relaxed);
        let fps = if seconds > 1.0 { frames as f64 / seconds } else { 0.0 };
        format!("metal present: {} frames in {:.3}s ({:.1} fps)", frames, seconds, fps)
    }
}

impl Drop for MetalPresenter {
    fn drop(&mut self) {
        self.jobs.take(); // the channel closes, the worker finishes its job
    }
}
