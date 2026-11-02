//! PICA resource adaptation, using the shared decode and filtering state.
use super::*;
use super::super::{BoundTexture, Wrap};
use super::super::pica_state::Filtering;
use objc2_metal::*;
use std::sync::Arc;

pub struct OwnedTextureBinding {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
}
impl OwnedTextureBinding {
    pub fn borrowed(&self) -> TextureBinding<'_> {
        TextureBinding { texture: &self.texture, sampler: &self.sampler }
    }
}
pub(super) struct Texture {
    image: Retained<ProtocolObject<dyn MTLTexture>>,
    texels: Arc<[[u8; 4]]>, width: u32, height: u32, levels: u32,
}
impl HardwareMetal {
    pub(crate) fn bind_pica_texture(&mut self, bound: &BoundTexture) -> Result<OwnedTextureBinding, String> {
        if bound.drawn.is_some() || bound.replacement.is_some() {
            return Err("unsupported Metal surface/texture-pack binding; coherent fallback required".into());
        }
        let filtering = Filtering::of(bound);
        if filtering.bias != 0 {
            return Err("unsupported Metal sampler LOD bias; coherent fallback required".into());
        }
        let (mut width, mut height) = (bound.width, bound.height);
        if width == 0 || height == 0 || bound.levels == 0 || bound.levels > 1 + width.max(height).ilog2() {
            return Err("invalid Metal texture dimensions/mipmap count".into());
        }
        let mut count = 0usize;
        for _ in 0..bound.levels {
            count = count.checked_add(width as usize * height as usize).ok_or("texture length overflow")?;
            width = (width / 2).max(1); height = (height / 2).max(1);
        }
        if bound.texels.len() != count { return Err("Metal mip chain length mismatch".into()); }
        let key = Arc::as_ptr(&bound.texels) as *const [u8; 4] as usize;
        let cached = self.pica_textures.get(&key).is_some_and(|entry|
            Arc::ptr_eq(&entry.texels, &bound.texels) && entry.width == bound.width && entry.height == bound.height && entry.levels == bound.levels);
        if !cached {
            let descriptor = MTLTextureDescriptor::new();
            descriptor.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
            descriptor.setStorageMode(MTLStorageMode::Managed);
            descriptor.setUsage(MTLTextureUsage::ShaderRead);
            unsafe { descriptor.setWidth(bound.width as usize); descriptor.setHeight(bound.height as usize); descriptor.setMipmapLevelCount(bound.levels as usize); }
            let image = self.device.newTextureWithDescriptor(&descriptor).ok_or("Metal texture allocation failed")?;
            let (mut width, mut height, mut offset) = (bound.width, bound.height, 0usize);
            for level in 0..bound.levels {
                let size = width as usize * height as usize;
                unsafe { image.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: width as usize, height: height as usize, depth: 1 } },
                    level as usize, NonNull::from(bound.texels[offset..offset+size].as_flattened()).cast(), width as usize * 4); }
                offset += size; width = (width/2).max(1); height = (height/2).max(1);
            }
            self.pica_textures.insert(key, Texture { image, texels: bound.texels.clone(), width: bound.width, height: bound.height, levels: bound.levels });
        }
        if !self.pica_samplers.contains_key(&filtering) {
            let descriptor = MTLSamplerDescriptor::new();
            let filter = |linear| if linear { MTLSamplerMinMagFilter::Linear } else { MTLSamplerMinMagFilter::Nearest };
            let address = |wrap| match wrap { Wrap::Repeat => MTLSamplerAddressMode::Repeat, Wrap::MirroredRepeat => MTLSamplerAddressMode::MirrorRepeat, _ => MTLSamplerAddressMode::ClampToEdge };
            descriptor.setMinFilter(filter(filtering.minify)); descriptor.setMagFilter(filter(filtering.magnify));
            descriptor.setMipFilter(if filtering.between { MTLSamplerMipFilter::Linear } else { MTLSamplerMipFilter::Nearest });
            descriptor.setSAddressMode(address(filtering.s)); descriptor.setTAddressMode(address(filtering.t));
            descriptor.setLodMinClamp(filtering.levels.0 as f32); descriptor.setLodMaxClamp(filtering.levels.1 as f32);
            let sampler = self.device.newSamplerStateWithDescriptor(&descriptor).ok_or("Metal sampler allocation failed")?;
            self.pica_samplers.insert(filtering, sampler);
        }
        Ok(OwnedTextureBinding { texture: self.pica_textures[&key].image.clone(), sampler: self.pica_samplers[&filtering].clone() })
    }
}


#[cfg(test)]
mod resource_tests {
    use super::*;
    #[test]
    fn pica_mips_are_owned_cached_and_validated() {
        let mut hardware=HardwareMetal::new().unwrap();
        let mut bound=BoundTexture { texels:Arc::from([vec![[255,0,0,255];256],vec![[0,255,0,255];64]].concat()),
            drawn:None,linear:false,wrap_s:Wrap::Repeat,wrap_t:Wrap::MirroredRepeat,width:16,height:16,
            border:[0.0;4],replacement:None,levels:2,lod:super::super::super::Lod::default() };
        let first=hardware.bind_pica_texture(&bound).unwrap();
        let second=hardware.bind_pica_texture(&bound).unwrap();
        assert_eq!(Retained::as_ptr(&first.texture),Retained::as_ptr(&second.texture));
        assert_eq!(first.texture.mipmapLevelCount(),2);
        assert!(Arc::strong_count(&bound.texels)>1);
        let mut pixel=[0u8;4];
        unsafe { first.texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(NonNull::from(&mut pixel[..]).cast(),4,
            MTLRegion {origin:MTLOrigin{x:0,y:0,z:0},size:MTLSize{width:1,height:1,depth:1}},1); }
        assert_eq!(pixel,[0,255,0,255]);
        bound.lod.bias=256; assert!(hardware.bind_pica_texture(&bound).is_err());
        bound.lod.bias=0; bound.texels=Arc::from(vec![[0u8;4]]);
        assert!(hardware.bind_pica_texture(&bound).is_err());
    }
}
