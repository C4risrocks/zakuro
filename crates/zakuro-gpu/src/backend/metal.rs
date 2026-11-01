//! The Metal spike (M0): one window, one CAMetalLayer, a timed clear+present.
//!
//! The goal this phase answers with numbers: pipeline creation cost, the
//! per-frame present cost, and whether the CAMetalLayer glue over the raw
//! winit window is sane — before promising a full PICA raster backend.
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction,
    MTLCommandBuffer, MTLCommandQueue, MTLCommandEncoder as _, MTLDevice, MTLLibrary,
    MTLRenderCommandEncoder as _, MTLCreateSystemDefaultDevice,
};
use objc2_quartz_core::CAMetalLayer;

/// one presenter: device, queue, layer, and the trivial pipeline of the spike.
pub struct MetalPresenter {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub layer: Retained<CAMetalLayer>,
    scale: f64,
    frames: u64,
    started: std::time::Instant,
}

const SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float4 color [[user(locn0)]]; };
vertex VOut spike_vertex(uint vid [[vertex_id]]) {
    VOut out;
    float x = float((vid >> 1) & 1) * 4.0 - 1.0;
    float y = float(vid & 1) * 4.0 - 1.0;
    out.position = float4(x, y, 0.0, 1.0);
    out.color = float4(0.0, 0.35, 1.0, 1.0);
    return out;
}
fragment float4 spike_fragment(VOut in [[stage_in]]) {
    return in.color;
}
";

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

impl MetalPresenter {
    /// creates the device + layer and installs it into the window's NSView.
    /// the view arrives as the raw pointer, the crate keeps the objc2 types.
    pub fn new(ns_view: *mut core::ffi::c_void, width: u32, height: u32, scale: f64) -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or("no Metal device on this Mac")?;
        let queue = device.newCommandQueue().ok_or("no Metal command queue")?;

        let source = ns(SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|error| format!("the spike shader did not compile: {error}"))?;
        let vertex = library
            .newFunctionWithName(&ns("spike_vertex"))
            .ok_or("no spike_vertex function")?;
        let fragment = library
            .newFunctionWithName(&ns("spike_fragment"))
            .ok_or("no spike_fragment function")?;
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        let attachments = unsafe { descriptor.colorAttachments() };
        let attachment: Retained<objc2_metal::MTLRenderPipelineColorAttachmentDescriptor> =
            unsafe { attachments.objectAtIndexedSubscript(0) };
        attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        let pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| format!("no pipeline state: {error}"))?;

        // the layer, handed to the window's view
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&*device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setMaximumDrawableCount(3);
        let size = ((width * scale as u32) as f64, (height * scale as u32) as f64);
        layer.setDrawableSize(CGSize::new(size.0, size.1));
        unsafe {
            let view: &AnyObject = &*(ns_view as *const AnyObject);
            let _: () = objc2::msg_send![view, setLayer: &*layer];
            let _: () = objc2::msg_send![view, setWantsLayer: true];
        }
        Ok(MetalPresenter {
            device,
            queue,
            pipeline,
            layer,
            scale,
            frames: 0,
            started: std::time::Instant::now(),
        })
    }

    pub fn name(&self) -> &'static str {
        "Metal"
    }

    pub fn resize(&self, width: u32, height: u32) {
        let size = ((width.max(1) * self.scale as u32) as f64,
                    (height.max(1) * self.scale as u32) as f64);
        self.layer.setDrawableSize(CGSize::new(size.0, size.1));
    }

    /// the spike's frame: a giant triangle in a fixed hue, presented.
    pub fn present_spike_frame(&mut self, red_shift: f64) {
        let Some(drawable) = self.layer.nextDrawable() else { return };
        let pass = MTLRenderPassDescriptor::new();
        let attachments = pass.colorAttachments();
        let attachment: Retained<objc2_metal::MTLRenderPassColorAttachmentDescriptor> =
            unsafe { attachments.objectAtIndexedSubscript(0) };
        use objc2_quartz_core::CAMetalDrawable;
        attachment.setTexture(Some(drawable.texture().as_ref()));
        attachment.setLoadAction(MTLLoadAction::Clear);
        attachment.setStoreAction(MTLStoreAction::Store);
        let green = ((self.frames % 90) as f64 / 90.0).max(0.15);
        attachment.setClearColor(MTLClearColor { red: 0.05 + red_shift, green, blue: 0.25, alpha: 1.0 });
        if let Some(buffer) = self.queue.commandBuffer() {
            unsafe {
                if let Some(encoder) = buffer.renderCommandEncoderWithDescriptor(&pass) {
                    encoder.setRenderPipelineState(&self.pipeline);
                    unsafe {
                        encoder.drawPrimitives_vertexStart_vertexCount(
                            objc2_metal::MTLPrimitiveType::Triangle,
                            0,
                            3,
                        );
                    }
                    encoder.endEncoding();
                }
                buffer.commit();
            }
        }
        self.frames += 1;
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// the M0 report, in the format tests and docs already know.
    pub fn report(&self) -> String {
        let seconds = self.started.elapsed().as_secs_f64();
        let fps = if self.frames > 1 { (self.frames - 1) as f64 / seconds } else { 0.0 };
        format!("metal spike complete: {} frames in {:.3}s ({:.1} fps)", self.frames, seconds, fps)
    }
}
