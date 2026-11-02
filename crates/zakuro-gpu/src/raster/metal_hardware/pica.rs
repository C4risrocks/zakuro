//! PICA resource adaptation, using the shared decode and filtering state.
use super::super::pica_state::Filtering;
use super::super::{BoundTexture, Wrap};
use super::*;
use objc2_metal::*;
use std::sync::Arc;

pub struct OwnedTextureBinding {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
}
impl OwnedTextureBinding {
    pub fn borrowed(&self) -> TextureBinding<'_> {
        TextureBinding {
            texture: &self.texture,
            sampler: &self.sampler,
        }
    }
}
pub(super) struct Texture {
    image: Retained<ProtocolObject<dyn MTLTexture>>,
    texels: Arc<[[u8; 4]]>,
    width: u32,
    height: u32,
    levels: u32,
}
impl HardwareMetal {
    pub(crate) fn bind_pica_texture(
        &mut self,
        bound: &BoundTexture,
    ) -> Result<OwnedTextureBinding, String> {
        if bound.drawn.is_some() || bound.replacement.is_some() {
            return Err(
                "unsupported Metal surface/texture-pack binding; coherent fallback required".into(),
            );
        }
        let filtering = Filtering::of(bound);
        if filtering.bias != 0 {
            return Err("unsupported Metal sampler LOD bias; coherent fallback required".into());
        }
        let (mut width, mut height) = (bound.width, bound.height);
        if width == 0
            || height == 0
            || bound.levels == 0
            || bound.levels > 1 + width.max(height).ilog2()
        {
            return Err("invalid Metal texture dimensions/mipmap count".into());
        }
        let mut count = 0usize;
        for _ in 0..bound.levels {
            count = count
                .checked_add(width as usize * height as usize)
                .ok_or("texture length overflow")?;
            width = (width / 2).max(1);
            height = (height / 2).max(1);
        }
        if bound.texels.len() != count {
            return Err("Metal mip chain length mismatch".into());
        }
        let key = Arc::as_ptr(&bound.texels) as *const [u8; 4] as usize;
        let cached = self.pica_textures.get(&key).is_some_and(|entry| {
            Arc::ptr_eq(&entry.texels, &bound.texels)
                && entry.width == bound.width
                && entry.height == bound.height
                && entry.levels == bound.levels
        });
        if !cached {
            let descriptor = MTLTextureDescriptor::new();
            descriptor.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
            descriptor.setStorageMode(MTLStorageMode::Managed);
            descriptor.setUsage(MTLTextureUsage::ShaderRead);
            unsafe {
                descriptor.setWidth(bound.width as usize);
                descriptor.setHeight(bound.height as usize);
                descriptor.setMipmapLevelCount(bound.levels as usize);
            }
            let image = self
                .device
                .newTextureWithDescriptor(&descriptor)
                .ok_or("Metal texture allocation failed")?;
            let (mut width, mut height, mut offset) = (bound.width, bound.height, 0usize);
            for level in 0..bound.levels {
                let size = width as usize * height as usize;
                unsafe {
                    image.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                        MTLRegion {
                            origin: MTLOrigin { x: 0, y: 0, z: 0 },
                            size: MTLSize {
                                width: width as usize,
                                height: height as usize,
                                depth: 1,
                            },
                        },
                        level as usize,
                        NonNull::from(bound.texels[offset..offset + size].as_flattened()).cast(),
                        width as usize * 4,
                    );
                }
                offset += size;
                width = (width / 2).max(1);
                height = (height / 2).max(1);
            }
            self.pica_textures.insert(
                key,
                Texture {
                    image,
                    texels: bound.texels.clone(),
                    width: bound.width,
                    height: bound.height,
                    levels: bound.levels,
                },
            );
        }
        if !self.pica_samplers.contains_key(&filtering) {
            let descriptor = MTLSamplerDescriptor::new();
            let filter = |linear| {
                if linear {
                    MTLSamplerMinMagFilter::Linear
                } else {
                    MTLSamplerMinMagFilter::Nearest
                }
            };
            let address = |wrap| match wrap {
                Wrap::Repeat => MTLSamplerAddressMode::Repeat,
                Wrap::MirroredRepeat => MTLSamplerAddressMode::MirrorRepeat,
                _ => MTLSamplerAddressMode::ClampToEdge,
            };
            descriptor.setMinFilter(filter(filtering.minify));
            descriptor.setMagFilter(filter(filtering.magnify));
            descriptor.setMipFilter(if filtering.between {
                MTLSamplerMipFilter::Linear
            } else {
                MTLSamplerMipFilter::Nearest
            });
            descriptor.setSAddressMode(address(filtering.s));
            descriptor.setTAddressMode(address(filtering.t));
            descriptor.setLodMinClamp(filtering.levels.0 as f32);
            descriptor.setLodMaxClamp(filtering.levels.1 as f32);
            let sampler = self
                .device
                .newSamplerStateWithDescriptor(&descriptor)
                .ok_or("Metal sampler allocation failed")?;
            self.pica_samplers.insert(filtering, sampler);
        }
        Ok(OwnedTextureBinding {
            texture: self.pica_textures[&key].image.clone(),
            sampler: self.pica_samplers[&filtering].clone(),
        })
    }
}

#[derive(Clone, Copy, Default)]
pub struct RasterState {
    pub key: PipelineKey,
    pub reference: u32,
    pub blend_color: [u8; 4],
    /// Top-left target pixels: x, y, width, height.
    pub scissor: Option<[u32; 4]>,
}
impl RasterState {
    pub(crate) fn from_pica(
        r: &[u32],
        depth: bool,
        lighting: bool,
        procedural: bool,
        writes_depth: bool,
    ) -> Result<Self, String> {
        use crate::registers::*;
        if r.len() < 0x300 {
            return Err("PICA register block truncated".into());
        }
        let mut key = PipelineKey {
            depth,
            ..PipelineKey::default()
        };
        key.mask = if r[REG_COLOR_BUFFER_WRITE] != 0 {
            (r[REG_DEPTH_COLOR_MASK] >> 8) & 15
        } else {
            0
        };
        key.blend = (r[REG_COLOR_OPERATION] & 0x100 != 0).then_some(r[REG_BLEND_FUNC] & 0xFFFF0707);
        if key.blend.is_none() {
            match crate::blend::LogicOp::read(r) {
                None | Some(crate::blend::LogicOp::Copy) => {}
                Some(crate::blend::LogicOp::Noop) => key.mask = 0,
                _ => {
                    return Err(
                        "unsupported Metal logic operation; coherent software fallback required"
                            .into(),
                    );
                }
            }
        }
        let writable = r[REG_DEPTH_STENCIL_WRITE] != 0;
        key.depth_control = r[REG_DEPTH_COLOR_MASK] & 0x71
            | if writable {
                r[REG_DEPTH_COLOR_MASK] & 0x1000
            } else {
                0
            };
        if depth && r[REG_DEPTH_BUFFER_FORMAT] & 3 == 3 {
            key.stencil_test = r[REG_STENCIL_TEST] & 0xFF00FFFF;
            key.stencil_op = r[REG_STENCIL_OP] & 0x777;
            if !writable {
                key.stencil_test &= !0xFF00;
            }
        }
        for (i, base) in [0xC0, 0xC8, 0xD0, 0xD8, 0xF0, 0xF8].into_iter().enumerate() {
            key.fragment[i] = r[base] & 0x0FFF0FFF;
            key.fragment[6 + i] = r[base + 1] & 0x00777FFF;
            key.fragment[12 + i] = r[base + 2] & 0x000F000F;
            key.fragment[18 + i] = r[base + 4] & 0x00030003;
        }
        key.fragment[24] = r[0xE0] & 0xFF00;
        key.fragment[25] = r[REG_TEXTURE_CONFIG] & 0x2307 | if procedural { 0x400 } else { 0 };
        key.fragment[26] = lighting as u32;
        key.fragment[27] = r[REG_ALPHA_TEST] & 0x71;
        key.fragment[28] = if writes_depth { 0x100 } else { 0 };
        Ok(Self {
            key,
            reference: (r[REG_STENCIL_TEST] >> 16) & 255,
            blend_color: r[0x103].to_le_bytes(),
            scissor: None,
        })
    }
}
fn compare(value: u32) -> MTLCompareFunction {
    [
        MTLCompareFunction::Never,
        MTLCompareFunction::Always,
        MTLCompareFunction::Equal,
        MTLCompareFunction::NotEqual,
        MTLCompareFunction::Less,
        MTLCompareFunction::LessEqual,
        MTLCompareFunction::Greater,
        MTLCompareFunction::GreaterEqual,
    ][(value & 7) as usize]
}
fn stencil(value: u32) -> MTLStencilOperation {
    [
        MTLStencilOperation::Keep,
        MTLStencilOperation::Zero,
        MTLStencilOperation::Replace,
        MTLStencilOperation::IncrementClamp,
        MTLStencilOperation::DecrementClamp,
        MTLStencilOperation::Invert,
        MTLStencilOperation::IncrementWrap,
        MTLStencilOperation::DecrementWrap,
    ][(value & 7) as usize]
}
impl HardwareMetal {
    pub(super) fn pica_depth_state(
        &mut self,
        key: &PipelineKey,
    ) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>, String> {
        if let Some(state) = self.depth_states.get(key) {
            return Ok(state.clone());
        }
        let d = MTLDepthStencilDescriptor::new();
        d.setDepthCompareFunction(if key.depth && key.depth_control & 1 != 0 {
            compare(key.depth_control >> 4)
        } else {
            MTLCompareFunction::Always
        });
        d.setDepthWriteEnabled(key.depth && key.depth_control & 0x1000 != 0);
        if key.depth && (key.stencil_test & 1 != 0 || key.stencil_op != 0) {
            let s = MTLStencilDescriptor::new();
            s.setStencilCompareFunction(if key.stencil_test & 1 != 0 {
                compare(key.stencil_test >> 4)
            } else {
                MTLCompareFunction::Always
            });
            s.setReadMask(key.stencil_test >> 24);
            s.setWriteMask((key.stencil_test >> 8) & 255);
            s.setStencilFailureOperation(stencil(key.stencil_op));
            s.setDepthFailureOperation(stencil(key.stencil_op >> 4));
            s.setDepthStencilPassOperation(stencil(key.stencil_op >> 8));
            d.setFrontFaceStencil(Some(&s));
            d.setBackFaceStencil(Some(&s));
        }
        let state = self
            .device
            .newDepthStencilStateWithDescriptor(&d)
            .ok_or("Metal depth/stencil state failed")?;
        self.depth_states.insert(*key, state.clone());
        Ok(state)
    }
}
pub(super) fn configure_blend(
    color: &MTLRenderPipelineColorAttachmentDescriptor,
    blend: Option<u32>,
) {
    color.setBlendingEnabled(blend.is_some());
    if let Some(raw) = blend {
        let equation = |v: u32| match v & 7 {
            1 => MTLBlendOperation::Subtract,
            2 => MTLBlendOperation::ReverseSubtract,
            3 => MTLBlendOperation::Min,
            4 => MTLBlendOperation::Max,
            _ => MTLBlendOperation::Add,
        };
        let factor = |v: u32| {
            [
                MTLBlendFactor::Zero,
                MTLBlendFactor::One,
                MTLBlendFactor::SourceColor,
                MTLBlendFactor::OneMinusSourceColor,
                MTLBlendFactor::DestinationColor,
                MTLBlendFactor::OneMinusDestinationColor,
                MTLBlendFactor::SourceAlpha,
                MTLBlendFactor::OneMinusSourceAlpha,
                MTLBlendFactor::DestinationAlpha,
                MTLBlendFactor::OneMinusDestinationAlpha,
                MTLBlendFactor::BlendColor,
                MTLBlendFactor::OneMinusBlendColor,
                MTLBlendFactor::BlendAlpha,
                MTLBlendFactor::OneMinusBlendAlpha,
                MTLBlendFactor::SourceAlphaSaturated,
                MTLBlendFactor::One,
            ][(v & 15) as usize]
        };
        color.setRgbBlendOperation(equation(raw));
        color.setAlphaBlendOperation(equation(raw >> 8));
        color.setSourceRGBBlendFactor(factor(raw >> 16));
        color.setDestinationRGBBlendFactor(factor(raw >> 20));
        color.setSourceAlphaBlendFactor(factor(raw >> 24));
        color.setDestinationAlphaBlendFactor(factor(raw >> 28));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pica_state_preserves_dynamic_values_and_rejects_logic_ops() {
        use crate::registers::*;
        let mut r = vec![0u32; 0x300];
        r[REG_LOGIC_OP] = 3;
        r[REG_COLOR_BUFFER_WRITE] = 1;
        r[REG_DEPTH_COLOR_MASK] = 0xF00;
        r[REG_DEPTH_BUFFER_FORMAT] = 3;
        r[REG_STENCIL_TEST] = 1 | (2 << 4) | (1 << 16) | (255 << 24);
        let first = RasterState::from_pica(&r, true, true, true, true).unwrap();
        assert_eq!(first.key.fragment[26], 1);
        assert_eq!(first.key.fragment[25] & 0x400, 0x400);
        assert_eq!(first.key.fragment[28] & 0x100, 0x100);
        r[REG_STENCIL_TEST] ^= 3 << 16;
        let second = RasterState::from_pica(&r, true, true, true, true).unwrap();
        assert_eq!(first.key, second.key, "stencil reference is dynamic");
        assert_ne!(first.reference, second.reference);
        r[REG_LOGIC_OP] = 6;
        assert_eq!(
            RasterState::from_pica(&r, true, false, false, false)
                .unwrap()
                .key
                .mask,
            0
        );
        r[REG_LOGIC_OP] = 11;
        assert!(RasterState::from_pica(&r, true, false, false, false).is_err());
        assert!(RasterState::from_pica(&[], false, false, false, false).is_err());
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    #[test]
    fn pica_mips_are_owned_cached_and_validated() {
        let mut hardware = HardwareMetal::new().unwrap();
        let mut bound = BoundTexture {
            texels: Arc::from([vec![[255, 0, 0, 255]; 256], vec![[0, 255, 0, 255]; 64]].concat()),
            drawn: None,
            linear: false,
            wrap_s: Wrap::Repeat,
            wrap_t: Wrap::MirroredRepeat,
            width: 16,
            height: 16,
            border: [0.0; 4],
            replacement: None,
            levels: 2,
            lod: super::super::super::Lod::default(),
        };
        let first = hardware.bind_pica_texture(&bound).unwrap();
        let second = hardware.bind_pica_texture(&bound).unwrap();
        assert_eq!(
            Retained::as_ptr(&first.texture),
            Retained::as_ptr(&second.texture)
        );
        assert_eq!(first.texture.mipmapLevelCount(), 2);
        assert!(Arc::strong_count(&bound.texels) > 1);
        let mut pixel = [0u8; 4];
        unsafe {
            first.texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::from(&mut pixel[..]).cast(),
                4,
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                },
                1,
            );
        }
        assert_eq!(pixel, [0, 255, 0, 255]);
        bound.lod.bias = 256;
        assert!(hardware.bind_pica_texture(&bound).is_err());
        bound.lod.bias = 0;
        bound.texels = Arc::from(vec![[0u8; 4]]);
        assert!(hardware.bind_pica_texture(&bound).is_err());
    }
}
