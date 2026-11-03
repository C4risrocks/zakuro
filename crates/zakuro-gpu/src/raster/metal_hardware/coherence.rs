//! Synchronous, write-through guest surfaces. Guest memory is canonical after
//! every successful draw; H6 may replace this policy with fenced dirty ranges.
use super::*;
use crate::GpuMemory;
use objc2_metal::*;

#[derive(Debug)]
pub enum CoherenceError {
    Unsupported(String),
    InvalidInput(String),
    Execution(String),
}
impl std::fmt::Display for CoherenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::Execution(s) => write!(f, "execution: {s}"),
        }
    }
}
impl std::error::Error for CoherenceError {}
#[derive(Clone, Copy)]
pub struct GuestTarget {
    pub addr: u32,
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub tiled: bool,
}
#[derive(Clone, Copy)]
pub struct GuestDepth {
    pub addr: u32,
    pub bytes: u32,
    pub tiled: bool,
}
pub struct GuestDraw<'a> {
    pub target: GuestTarget,
    pub depth: Option<GuestDepth>,
    pub scale: u32,
    pub vertices: &'a [f32],
    pub indices: &'a [u32],
    pub textures: [Option<TextureBinding<'a>>; 3],
    pub state: RasterState,
    pub uniforms: &'a [u32],
    pub tables: &'a [u32],
}
pub(super) fn overlaps(a: u32, an: u32, b: u32, bn: u32) -> bool {
    an != 0 && bn != 0 && (a as u64) < b as u64 + bn as u64 && (b as u64) < a as u64 + an as u64
}
pub(super) fn validate_surface(
    addr: u32,
    w: u32,
    h: u32,
    bpp: u32,
    tiled: bool,
) -> Result<u32, CoherenceError> {
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        return Err(CoherenceError::InvalidInput("surface dimensions".into()));
    }
    if tiled && (!w.is_multiple_of(8) || !h.is_multiple_of(8)) {
        return Err(CoherenceError::Unsupported("partial Morton tiles".into()));
    }
    let len = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(bpp))
        .ok_or_else(|| CoherenceError::InvalidInput("surface size overflow".into()))?;
    if addr as u64 + len as u64 > 1u64 << 32 {
        return Err(CoherenceError::InvalidInput(
            "surface address overflow".into(),
        ));
    }
    Ok(len)
}
fn at(x: u32, y: u32, w: u32, bpp: u32, tiled: bool) -> usize {
    if tiled {
        crate::format::morton_offset(x, y, w, bpp) as usize
    } else {
        ((y * w + x) * bpp) as usize
    }
}
fn exec(s: impl Into<String>) -> CoherenceError {
    CoherenceError::Execution(s.into())
}

/// H6: the depth twin resident next to a color surface — the float image
/// plus its readback planes and the shadow of the guest's bytes the image
/// matched last. A draw stages only the bytes the guest moved since, and
/// the pixels land in guest memory at the draw's write-back (the
/// write-through the fences still keep).
pub(crate) struct MetalDepth {
    addr: u32,
    width: u32,
    height: u32,
    bytes: u32,
    tiled: bool,
    scale: u32,
    image: Retained<ProtocolObject<dyn MTLTexture>>,
    depth_plane: Retained<ProtocolObject<dyn MTLBuffer>>,
    stencil_plane: Retained<ProtocolObject<dyn MTLBuffer>>,
    dr: usize,
    sr: usize,
    shadow: Vec<u8>,
}

impl MetalDepth {
    /// the guest's bytes a whole depth surface takes.
    fn size(&self) -> u32 {
        self.width * self.height * self.bytes
    }
}
fn texture(
    device: &ProtocolObject<dyn MTLDevice>,
    w: u32,
    h: u32,
    format: MTLPixelFormat,
    private: bool,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, CoherenceError> {
    let d = MTLTextureDescriptor::new();
    d.setPixelFormat(format);
    d.setStorageMode(if private {
        MTLStorageMode::Private
    } else {
        MTLStorageMode::Managed
    });
    d.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    unsafe {
        d.setWidth(w as usize);
        d.setHeight(h as usize);
    }
    device
        .newTextureWithDescriptor(&d)
        .ok_or_else(|| exec("surface allocation failed"))
}
fn wait(buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), CoherenceError> {
    buffer.commit();
    buffer.waitUntilCompleted();
    if buffer.status() != MTLCommandBufferStatus::Completed {
        return Err(exec("Metal command failed; guest memory unchanged"));
    }
    Ok(())
}
impl HardwareMetal {
    /// Safe because every Metal write is committed synchronously to guest memory.
    pub(crate) fn sync_guest<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) {
        // the fences write the resident images back before they drop: the
        // guest keeps what the GPU drew, at every fence point
        let color_hits: Vec<usize> = (0..self.surfaces.len())
            .filter(|&i| overlaps(addr, len, self.surfaces[i].addr, self.surfaces[i].size()))
            .collect();
        for index in color_hits {
            if let Err(error) = self.write_back(memory, index) {
                log::error!("the Metal host could not write a surface back, {error}");
            }
        }
        self.surfaces
            .retain(|s| !overlaps(addr, len, s.addr, s.size()));
        let depth_hits: Vec<usize> = (0..self.depths.len())
            .filter(|&i| overlaps(addr, len, self.depths[i].addr, self.depths[i].size()))
            .collect();
        for index in depth_hits {
            if let Err(error) = self.finish_depth(memory, index) {
                log::error!("the Metal host could not write a depth image back, {error}");
            }
        }
        self.depths
            .retain(|s| !overlaps(addr, len, s.addr, s.size()));
        // Decoded identities may be reused after state restore or an external write.
        self.pica_textures.clear();
    }

    /// the depth's resident image for this draw: found or made, and its
    /// planes staged only when the guest's bytes moved since the image
    /// last matched (the shadow). returns the entry and whether the
    /// staged bytes need their upload blit in the draw's own buffer.
    fn depth_for<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        d: GuestDepth,
        nw: u32,
        nh: u32,
        scale: u32,
    ) -> Result<(usize, bool), CoherenceError> {
        let bytes = d.bytes;
        let len = (nw * nh * bytes) as usize;
        let mut fresh = vec![0u8; len];
        memory.read(d.addr, &mut fresh);
        let index = self.depths.iter().position(|entry| {
            entry.addr == d.addr
                && entry.width == nw
                && entry.height == nh
                && entry.bytes == bytes
                && entry.tiled == d.tiled
                && entry.scale == scale
        });
        let index = match index {
            Some(index) => index,
            None => {
                let image = texture(
                    &self.device,
                    nw * scale,
                    nh * scale,
                    MTLPixelFormat::Depth32Float_Stencil8,
                    true,
                )?;
                let (depth_plane, stencil_plane, dr, sr) =
                    self.depth_planes(nw * scale, nh * scale)?;
                self.depths.push(MetalDepth {
                    addr: d.addr,
                    width: nw,
                    height: nh,
                    bytes,
                    tiled: d.tiled,
                    scale,
                    image,
                    depth_plane,
                    stencil_plane,
                    dr,
                    sr,
                    shadow: Vec::new(),
                });
                self.depths.len() - 1
            }
        };
        if fresh != self.depths[index].shadow {
            let (w, h) = (nw * scale, nh * scale);
            let maximum = if bytes == 2 { 65535.0 } else { 16777215.0 };
            let (dr, sr) = (self.depths[index].dr, self.depths[index].sr);
            let depth_plane = self.depths[index].depth_plane.clone();
            let stencil_plane = self.depths[index].stencil_plane.clone();
            unsafe {
                let depths = std::slice::from_raw_parts_mut(
                    depth_plane.contents().as_ptr() as *mut u8,
                    dr * h as usize,
                );
                let stencils = std::slice::from_raw_parts_mut(
                    stencil_plane.contents().as_ptr() as *mut u8,
                    sr * h as usize,
                );
                for y in 0..h {
                    for x in 0..w {
                        let at = at(x / scale, y / scale, nw, bytes, d.tiled);
                        let mut raw = [0u8; 4];
                        raw[..bytes as usize].copy_from_slice(&fresh[at..at + bytes as usize]);
                        let value = u32::from_le_bytes(raw)
                            & if bytes == 2 { 65535 } else { 16777215 };
                        let row = (h - 1 - y) as usize;
                        depths[row * dr + x as usize * 4..row * dr + x as usize * 4 + 4]
                            .copy_from_slice(&(value as f32 / maximum).to_le_bytes());
                        stencils[row * sr + x as usize] =
                            if bytes == 4 { raw[3] } else { 0 };
                    }
                }
            }
            self.depths[index].shadow = fresh;
            Ok((index, true))
        } else {
            Ok((index, false))
        }
    }

    /// the depth's truth lands in guest memory: the planes (already filled
    /// by the draw's own readback blits on the waited command buffer) are
    /// decoded at the samples the GPU changed, the exact guest bits of the
    /// rest kept. the shadow then matches what memory holds.
    fn finish_depth<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        index: usize,
    ) -> Result<(), CoherenceError> {
        let (addr, bytes, tiled, scale) = {
            let entry = &self.depths[index];
            (entry.addr, entry.bytes, entry.tiled, entry.scale)
        };
        let (nw, nh) = (self.depths[index].width, self.depths[index].height);
        let (dr, sr) = (self.depths[index].dr, self.depths[index].sr);
        let depth_plane = self.depths[index].depth_plane.clone();
        let stencil_plane = self.depths[index].stencil_plane.clone();
        let (w, h) = (nw * scale, nh * scale);
        let len = (nw * nh * bytes) as usize;
        let mut before = vec![0u8; len];
        memory.read(addr, &mut before);
        let mut out = before.clone();
        let maximum = if bytes == 2 { 65535.0 } else { 16777215.0 };
        unsafe {
            let depths = std::slice::from_raw_parts(
                depth_plane.contents().as_ptr() as *const u8,
                dr * h as usize,
            );
            let stencils = std::slice::from_raw_parts(
                stencil_plane.contents().as_ptr() as *const u8,
                sr * h as usize,
            );
            for y in 0..nh {
                for x in 0..nw {
                    let at = at(x, y, nw, bytes, tiled);
                    let row = (h - 1 - (y * scale + scale / 2)) as usize;
                    let col = (x * scale + scale / 2) as usize;
                    let value = f32::from_le_bytes(
                        depths[row * dr + col * 4..row * dr + col * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let mut raw = [0u8; 4];
                    raw[..bytes as usize].copy_from_slice(&before[at..at + bytes as usize]);
                    let previous =
                        u32::from_le_bytes(raw) & if bytes == 2 { 65535 } else { 16777215 };
                    // Preserve exact guest bits if the GPU did not change the sample.
                    if value.to_bits() != (previous as f32 / maximum).to_bits() {
                        let encoded = (value.clamp(0.0, 1.0) * maximum) as u32;
                        let n = if bytes == 2 { 2 } else { 3 };
                        out[at..at + n].copy_from_slice(&encoded.to_le_bytes()[..n]);
                    }
                    if bytes == 4 {
                        out[at + 3] = stencils[row * sr + col];
                    }
                }
            }
        }
        memory.write(addr, &out);
        self.depths[index].shadow = out;
        Ok(())
    }
    pub(crate) fn forget_guest(&mut self) {
        self.surfaces.clear();
        self.depths.clear();
        self.pica_textures.clear();
        self.pica_samplers.clear();
        self.ring.live.clear();
        self.ring.head = 0;
        // Failed submissions require recreation, not a silent recovery on restore.
    }
    pub(crate) fn draw_guest<M: GpuMemory>(
        &mut self,
        memory: &mut M,
        draw: &GuestDraw<'_>,
        cache: &mut super::super::TextureCache,
    ) -> Result<(), CoherenceError> {
        if self.failed {
            return Err(exec("Metal backend requires recreation"));
        }
        if draw.state.key.vertex != VertexStage::Placed {
            return Err(CoherenceError::Unsupported("vertex shading mode".into()));
        }
        if draw.uniforms.len() < super::super::pica_state::DRAW_WORDS
            || !draw.vertices.len().is_multiple_of(24)
            || !draw.indices.len().is_multiple_of(3)
            || draw
                .indices
                .iter()
                .any(|i| *i as usize >= draw.vertices.len() / 24)
        {
            return Err(CoherenceError::InvalidInput("draw input layout".into()));
        }
        for unit in 0..3 {
            if draw.uniforms[51] & (1 << unit) != 0 && draw.textures[unit].is_none() {
                return Err(CoherenceError::InvalidInput(
                    "unbound active texture".into(),
                ));
            }
        }
        let t = draw.target;
        if !(1..=3).contains(&draw.scale) {
            return Err(CoherenceError::Unsupported(
                "resolution scale outside 1..3".into(),
            ));
        }
        let len = validate_surface(
            t.addr,
            t.width,
            t.height,
            t.format.bytes_per_pixel() as u32,
            t.tiled,
        )?;
        if draw.state.key.depth != draw.depth.is_some() {
            return Err(CoherenceError::InvalidInput("depth key mismatch".into()));
        }
        let depth_len = if let Some(d) = draw.depth {
            if !matches!(d.bytes, 2 | 3 | 4) {
                return Err(CoherenceError::InvalidInput("depth format".into()));
            }
            let n = validate_surface(d.addr, t.width, t.height, d.bytes, d.tiled)?;
            if overlaps(t.addr, len, d.addr, n) {
                return Err(CoherenceError::Unsupported("color/depth alias".into()));
            }
            Some(n)
        } else {
            None
        };
        let (w, h) = (t.width * draw.scale, t.height * draw.scale);
        if w > 8192 || h > 8192 {
            return Err(CoherenceError::Unsupported(
                "scaled surface too large".into(),
            ));
        }
        if let Some([x, y, rw, rh]) = draw.state.scissor {
            if x.checked_add(rw).is_none_or(|n| n > t.width)
                || y.checked_add(rh).is_none_or(|n| n > t.height)
            {
                return Err(CoherenceError::InvalidInput("draw scissor".into()));
            }
            if rw == 0 || rh == 0 {
                return Ok(());
            }
        }
        if draw.indices.is_empty() {
            return Ok(());
        }
        // H6: the color target mounts as a resident surface (the honest
        // shadow scheme) and the depth twin as a resident float image. the
        // frame's draws land into them, and every draw's write-back keeps
        // the fence honest — guest memory canonical after every draw —
        // without the per-draw texture and plane allocations, and with a
        // single command buffer for the staged depth blits, the render
        // pass, the depth readback and the color's Managed sync (one wait).
        let color_index = self
            .check_surface_at(memory, t.addr, (t.width, t.height), t.format, t.tiled, draw.scale)
            .map_err(exec)?;
        let color = self.surfaces[color_index].texture.clone();
        let depth_index = match draw.depth {
            Some(d) => Some(self.depth_for(memory, d, t.width, t.height, draw.scale)?),
            None => None,
        };

        // the float twin and the scaled scissor carry into the native draw
        let mut state = draw.state;
        state.key.float_depth = true;
        if let Some(rect) = state.scissor.as_mut() {
            for v in rect {
                *v = v
                    .checked_mul(draw.scale)
                    .ok_or_else(|| CoherenceError::InvalidInput("scissor overflow".into()))?;
            }
        }
        let depth_attachment = depth_index
            .map(|(index, _)| index)
            .map(|index| self.depths[index].image.clone());
        let native = PlacedDraw {
            vertices: draw.vertices,
            indices: draw.indices,
            textures: draw.textures,
            target: &color,
            clear: false,
            depth: depth_attachment.as_deref(),
            width: w,
            height: h,
        };

        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| exec("draw command allocation"))?;
        // the staged depth's upload blits, only where the shadow moved
        if let Some((index, true)) = depth_index {
            let image = self.depths[index].image.clone();
            let depth_plane = self.depths[index].depth_plane.clone();
            let stencil_plane = self.depths[index].stencil_plane.clone();
            let (dr, sr) = (self.depths[index].dr, self.depths[index].sr);
            let blit = cb
                .blitCommandEncoder()
                .ok_or_else(|| exec("depth encoder allocation"))?;
            unsafe {
                for (buffer, row, option) in [
                    (&depth_plane, dr, MTLBlitOption::DepthFromDepthStencil),
                    (&stencil_plane, sr, MTLBlitOption::StencilFromDepthStencil),
                ] {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                        buffer,
                        0,
                        row,
                        row * h as usize,
                        MTLSize { width: w as usize, height: h as usize, depth: 1 },
                        &image,
                        0,
                        0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                        option,
                    );
                }
            }
            {
                use objc2_metal::MTLCommandEncoder as _;
                blit.endEncoding();
            }
        }
        self.encode_placed(&cb, &native, &state, draw.uniforms, draw.tables)
            .map_err(exec)?;
        // the depth's truth and the color's Managed sync, on the same buffer
        {
            let blit = cb
                .blitCommandEncoder()
                .ok_or_else(|| exec("readback encoder allocation"))?;
            if let Some((index, _)) = depth_index {
                let image = self.depths[index].image.clone();
                let depth_plane = self.depths[index].depth_plane.clone();
                let stencil_plane = self.depths[index].stencil_plane.clone();
                let (dr, sr) = (self.depths[index].dr, self.depths[index].sr);
                unsafe {
                    for (buffer, row, option) in [
                        (&depth_plane, dr, MTLBlitOption::DepthFromDepthStencil),
                        (&stencil_plane, sr, MTLBlitOption::StencilFromDepthStencil),
                    ] {
                        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                            &image,
                            0,
                            0,
                            MTLOrigin { x: 0, y: 0, z: 0 },
                            MTLSize { width: w as usize, height: h as usize, depth: 1 },
                            buffer,
                            0,
                            row,
                            row * h as usize,
                            option,
                        );
                    }
                }
            }
            unsafe { blit.synchronizeResource((*color).as_ref()); }
            {
                use objc2_metal::MTLCommandEncoder as _;
                blit.endEncoding();
            }
        }
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            self.failed = true;
            return Err(exec("Metal draw failed; guest memory unchanged"));
        }
        self.ring.head = 0;
        self.ring.live.clear();

        // the readbacks run after the one wait: the color re-encoded into
        // the surface's own bytes, the depth's planes decoded at the
        // changed samples, both into guest memory (write-through)
        let rgba = self.readback_rgba(&cb, color_index).map_err(exec)?;
        self.write_back_rgba(memory, color_index, &rgba)
            .map_err(exec)?;
        if let Some((index, _)) = depth_index {
            self.finish_depth(memory, index)?;
        }
        cache.invalidate_range(t.addr, len);
        if let (Some(d), Some(n)) = (draw.depth, depth_len) {
            cache.invalidate_range(d.addr, n);
        }
        Ok(())
    }

    fn depth_planes(
        &self,
        w: u32,
        h: u32,
    ) -> Result<
        (
            Retained<ProtocolObject<dyn MTLBuffer>>,
            Retained<ProtocolObject<dyn MTLBuffer>>,
            usize,
            usize,
        ),
        CoherenceError,
    > {
        let dr = (w as usize * 4).next_multiple_of(256);
        let sr = (w as usize).next_multiple_of(256);
        let d = unsafe {
            self.device
                .newBufferWithLength_options(dr * h as usize, MTLResourceOptions::StorageModeShared)
        }
        .ok_or_else(|| exec("depth staging allocation"))?;
        let s = unsafe {
            self.device
                .newBufferWithLength_options(sr * h as usize, MTLResourceOptions::StorageModeShared)
        }
        .ok_or_else(|| exec("stencil staging allocation"))?;
        unsafe {
            std::ptr::write_bytes(d.contents().as_ptr() as *mut u8, 0, dr * h as usize);
            std::ptr::write_bytes(s.contents().as_ptr() as *mut u8, 0, sr * h as usize);
        }
        Ok((d, s, dr, sr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Memory(Vec<u8>);
    impl GpuMemory for Memory {
        fn read(&mut self, a: u32, out: &mut [u8]) {
            out.copy_from_slice(&self.0[a as usize..a as usize + out.len()]);
        }
        fn write(&mut self, a: u32, bytes: &[u8]) {
            self.0[a as usize..a as usize + bytes.len()].copy_from_slice(bytes);
        }
    }
    fn vertices() -> Vec<f32> {
        let mut out = Vec::new();
        for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            let mut v = [0.0; 24];
            v[0] = x;
            v[1] = y;
            v[3] = 1.0;
            v[4] = 1.0;
            v[7] = 1.0;
            v[14] = 0.5;
            out.extend(v);
        }
        out
    }
    fn uniforms() -> Vec<u32> {
        let mut u = vec![0; 408];
        for stage in 0..6 {
            u[stage * 8] = 0xF000F;
        }
        u
    }
    #[test]
    fn surface_ranges_reject_wraparound_and_partial_tiles() {
        assert!(!overlaps(0, 0, 0, 16));
        assert!(!overlaps(0, 16, 16, 16));
        assert!(overlaps(0, 17, 16, 16));
        assert!(matches!(
            validate_surface(u32::MAX - 3, 8, 8, 4, false),
            Err(CoherenceError::InvalidInput(_))
        ));
        assert!(matches!(
            validate_surface(0, 9, 8, 4, true),
            Err(CoherenceError::Unsupported(_))
        ));
        assert!(validate_surface(0, 0, 8, 4, false).is_err());
    }
    #[test]
    fn guest_color_depth_partial_draws_are_coherent_at_three_scales() {
        let mut hardware = HardwareMetal::new().unwrap();
        // One draw exactly fills this ring; the next requires retirement.
        hardware.ring = Ring::new(hardware.device.as_ref(), 2048).unwrap();
        let mut cache = super::super::super::TextureCache::default();
        let vertices = vertices();
        let uniforms = uniforms();
        for format in [
            ColorFormat::Rgba8,
            ColorFormat::Rgb8,
            ColorFormat::Rgb565,
            ColorFormat::Rgb5A1,
            ColorFormat::Rgba4,
        ] {
            for scale in 1..=3 {
                for bpp in [2, 3, 4] {
                    for tiled in [false, true] {
                        let mut memory = Memory(vec![0; 0x20000]);
                        let target = GuestTarget {
                            addr: 0x1000,
                            width: 16,
                            height: 16,
                            format,
                            tiled,
                        };
                        let depth = GuestDepth {
                            addr: 0x9000,
                            bytes: bpp,
                            tiled,
                        };
                        for y in 0..16 {
                            for x in 0..16 {
                                let p = target.addr as usize
                                    + at(x, y, 16, format.bytes_per_pixel() as u32, tiled);
                                target.format.encode(
                                    [0, 255, 0, 255],
                                    &mut memory.0[p..p + format.bytes_per_pixel()],
                                );
                            }
                        }
                        memory.0[0x9000..0x9000 + (256 * bpp) as usize].fill(255);
                        let mut state = RasterState::default();
                        state.key.depth = true;
                        state.key.depth_control = 0x1000;
                        if bpp == 4 {
                            state.key.stencil_test = 0xFF00;
                            state.key.stencil_op = 2 << 8;
                            state.reference = 7;
                        }
                        state.scissor = Some([4, 4, 8, 8]);
                        let input = GuestDraw {
                            target,
                            depth: Some(depth),
                            scale,
                            vertices: &vertices,
                            indices: &[0, 1, 2, 2, 1, 3],
                            textures: [None; 3],
                            state,
                            uniforms: &uniforms,
                            tables: &[],
                        };
                        hardware
                            .draw_guest(&mut memory, &input, &mut cache)
                            .unwrap();
                        assert_eq!(hardware.ring.head, 0, "completed draw retires staging");
                        for y in 0..16 {
                            for x in 0..16 {
                                let inside = (4..12).contains(&x) && (4..12).contains(&y);
                                let p = target.addr as usize
                                    + at(x, y, 16, format.bytes_per_pixel() as u32, tiled);
                                assert_eq!(
                                    target
                                        .format
                                        .decode(&memory.0[p..p + format.bytes_per_pixel()]),
                                    if inside {
                                        [255, 0, 0, 255]
                                    } else {
                                        [0, 255, 0, 255]
                                    },
                                    "scale {scale}, depth {bpp}, tiled {tiled}, pixel {x},{y}"
                                );
                                let p = depth.addr as usize + at(x, y, 16, bpp, tiled);
                                let mut raw = [0u8; 4];
                                raw[..bpp as usize].copy_from_slice(&memory.0[p..p + bpp as usize]);
                                let maximum = if bpp == 2 { 65535 } else { 16777215 };
                                assert_eq!(
                                    u32::from_le_bytes(raw) & maximum,
                                    if inside { maximum / 2 } else { maximum }
                                );
                                if bpp == 4 {
                                    assert_eq!(raw[3], if inside { 7 } else { 255 });
                                }
                            }
                        }
                        // CPU writes are preserved on the next draw, even without notification.
                        target.format.encode(
                            [0, 0, 255, 255],
                            &mut memory.0[0x1000..0x1000 + format.bytes_per_pixel()],
                        );
                        hardware
                            .draw_guest(&mut memory, &input, &mut cache)
                            .unwrap();
                        assert_eq!(
                            target
                                .format
                                .decode(&memory.0[0x1000..0x1000 + format.bytes_per_pixel()]),
                            [0, 0, 255, 255]
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn large_transfers_match_cpu_with_conversion_flip_and_partial_rows() {
        let mut hardware = HardwareMetal::new().unwrap();
        for format in [
            ColorFormat::Rgba8,
            ColorFormat::Rgb8,
            ColorFormat::Rgb565,
            ColorFormat::Rgb5A1,
            ColorFormat::Rgba4,
        ] {
            for flip in [false, true] {
                for tiled in [false, true] {
                    let mut gpu_memory = Memory(vec![0xA5; 0x30000]);
                    for y in 0..64 {
                        for x in 0..96 {
                            let p = 0x1000 + at(x, y, 96, 4, tiled);
                            ColorFormat::Rgba8.encode(
                                [x as u8, (y * 3) as u8, (x ^ y) as u8, 255],
                                &mut gpu_memory.0[p..p + 4],
                            );
                        }
                    }
                    let mut cpu_memory = Memory(gpu_memory.0.clone());
                    let transfer = super::super::super::Transfer {
                        input: 0x1000,
                        output: 0x12000,
                        input_width: 96,
                        output_width: 64,
                        output_height: 32,
                        copy: (48, 32),
                        scale: (2, 2),
                        flip,
                        input_linear: !tiled,
                        output_tiled: tiled,
                        input_format: ColorFormat::Rgba8,
                        output_format: format,
                    };
                    let flags = flip as u32
                        | ((!tiled as u32) << 1)
                        | ((format as u32) << 12)
                        | (2 << 24)
                        | (1 << 5);
                    crate::Gpu::new().display_transfer(
                        &mut cpu_memory,
                        0x1000,
                        0x12000,
                        96 | (64 << 16),
                        128 | (64 << 16),
                        flags,
                    );
                    hardware
                        .display_transfer(&mut gpu_memory, &transfer)
                        .unwrap();
                    let difference = gpu_memory
                        .0
                        .iter()
                        .zip(&cpu_memory.0)
                        .position(|(a, b)| a != b);
                    assert_eq!(
                        difference, None,
                        "format {format:?}, flip {flip}, tiled {tiled}"
                    );
                }
            }
        }
    }
    #[test]
    fn fills_copies_restore_and_overlapping_views_use_guest_memory() {
        let mut gpu = crate::Gpu::new();
        gpu.resources.metal_hardware = Some(HardwareMetal::new().unwrap());
        let mut memory = Memory(vec![0; 0x20000]);
        let mut cache = super::super::super::TextureCache::default();
        let target = GuestTarget {
            addr: 0x1000,
            width: 16,
            height: 16,
            format: ColorFormat::Rgba8,
            tiled: false,
        };
        let vertices = vertices();
        let u = uniforms();
        let input = GuestDraw {
            target,
            depth: None,
            scale: 1,
            vertices: &vertices,
            indices: &[0, 1, 2, 2, 1, 3],
            textures: [None; 3],
            state: RasterState::default(),
            uniforms: &u,
            tables: &[],
        };
        gpu.resources
            .metal_hardware
            .as_mut()
            .unwrap()
            .draw_guest(&mut memory, &input, &mut cache)
            .unwrap();
        assert_eq!(
            target.format.decode(&memory.0[0x1000..0x1004]),
            [255, 0, 0, 255]
        );
        // A buffer just drawn is decoded through the common TextureCache.
        let mut texture_registers = vec![0u32; 0x300];
        texture_registers[crate::registers::REG_TEXTURE_CONFIG] = 1;
        texture_registers[0x82] = (16 << 16) | 16;
        texture_registers[0x85] = target.addr / 8;

        let sampled = super::super::super::bind_texture(
            &texture_registers,
            &mut memory,
            &mut cache,
            0,
            None,
            false,
        )
        .unwrap();
        let hw = gpu.resources.metal_hardware.as_mut().unwrap();
        let binding = hw.bind_pica_texture(&sampled).unwrap();
        let mut textured_uniforms = uniforms();
        textured_uniforms[0] = 3 | (3 << 16);
        textured_uniforms[51] = 1;
        let second = GuestDraw {
            target: GuestTarget {
                addr: 0x3000,
                ..target
            },
            textures: [Some(binding.borrowed()), None, None],
            uniforms: &textured_uniforms,
            ..input
        };
        hw.draw_guest(&mut memory, &second, &mut cache).unwrap();
        assert_eq!(
            target.format.decode(&memory.0[0x3000..0x3004]),
            [255, 0, 0, 255],
            "draw to common texture to second draw"
        );
        let mut blue_vertices = vertices.clone();
        for v in blue_vertices.chunks_exact_mut(24) {
            v[4] = 0.0;
            v[6] = 1.0;
        }
        let blue = GuestDraw {
            vertices: &blue_vertices,
            ..input
        };
        hw.draw_guest(&mut memory, &blue, &mut cache).unwrap();
        let refreshed = super::super::super::bind_texture(
            &texture_registers,
            &mut memory,
            &mut cache,
            0,
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            refreshed.texels[0],
            [0, 0, 255, 255],
            "same-list second write invalidates decoded texture"
        );
        let snapshot = memory.0.clone();
        let registers = gpu.save_state();
        gpu.resources
            .metal_hardware
            .as_mut()
            .unwrap()
            .check_surface(&mut memory, 0x1000, (16, 16), target.format, false)
            .unwrap();
        let _ = super::super::super::bind_texture(
            &texture_registers,
            &mut memory,
            &mut gpu.resources.textures,
            0,
            None,
            false,
        )
        .unwrap();
        gpu.memory_fill(&mut memory, 0x1000, 0x1400, 0xFF00FF00, 4);
        // H6: the fence drops the surface the fill touched; the frame's
        // other residents stay (their own fences drop them)
        assert!(
            gpu.resources
                .metal_hardware
                .as_ref()
                .unwrap()
                .surfaces
                .iter()
                .all(|s| !overlaps(0x1000, 0x1400 - 0x1000, s.addr, s.size())),
            "the fill's fence drops the surface it touched"
        );
        let filled = super::super::super::bind_texture(
            &texture_registers,
            &mut memory,
            &mut gpu.resources.textures,
            0,
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            filled.texels[0],
            target.format.decode(&memory.0[0x1000..0x1004]),
            "fill invalidates same-list decoded data"
        );
        gpu.texture_copy(&mut memory, 0x1000, 0x3000, 1024, 0, 0);
        assert_eq!(&memory.0[0x1000..0x1400], &memory.0[0x3000..0x3400]);
        memory.0 = snapshot;
        gpu.load_state(&registers).unwrap();
        let hw = gpu.resources.metal_hardware.as_mut().unwrap();
        assert!(hw.surfaces.is_empty());
        hw.check_surface(&mut memory, 0x1000, (16, 16), target.format, false)
            .unwrap();
        hw.check_surface(&mut memory, 0x1040, (8, 8), target.format, false)
            .unwrap();
        hw.sync_guest(&mut memory, 0x1050, 1);
        assert!(hw.surfaces.is_empty(), "both overlapping views invalidated");
        let transfer = super::super::super::Transfer {
            input: 0x1000,
            output: 0x5000,
            input_width: 16,
            output_width: 16,
            output_height: 16,
            copy: (16, 16),
            scale: (1, 1),
            flip: false,
            input_linear: true,
            output_tiled: false,
            input_format: target.format,
            output_format: target.format,
        };
        hw.display_transfer(&mut memory, &transfer).unwrap();
        assert_eq!(&memory.0[0x1000..0x1400], &memory.0[0x5000..0x5400]);
        let alias = super::super::super::Transfer {
            output: 0x1040,
            ..transfer
        };
        assert!(hw.display_transfer(&mut memory, &alias).is_err());
        let before = memory.0.clone();
        let unsupported = GuestDraw { scale: 4, ..input };
        assert!(matches!(
            hw.draw_guest(&mut memory, &unsupported, &mut cache),
            Err(CoherenceError::Unsupported(_))
        ));
        let alias = GuestDraw {
            depth: Some(GuestDepth {
                addr: target.addr + 4,
                bytes: 4,
                tiled: false,
            }),
            state: RasterState {
                key: PipelineKey {
                    depth: true,
                    ..PipelineKey::default()
                },
                ..RasterState::default()
            },
            ..input
        };
        assert!(matches!(
            hw.draw_guest(&mut memory, &alias, &mut cache),
            Err(CoherenceError::Unsupported(_))
        ));
        hw.failed = true;
        assert!(matches!(
            hw.draw_guest(&mut memory, &input, &mut cache),
            Err(CoherenceError::Execution(_))
        ));
        hw.forget_guest();
        assert!(matches!(
            hw.draw_guest(&mut memory, &input, &mut cache),
            Err(CoherenceError::Execution(_))
        ));
        assert!(
            memory.0 == before,
            "rejected operations do not modify guest memory"
        );
        hw.failed = false;
        let mut invalid = input;
        invalid.target.addr = u32::MAX - 3;
        assert!(matches!(
            hw.draw_guest(&mut memory, &invalid, &mut cache),
            Err(CoherenceError::InvalidInput(_))
        ));
    }
}
