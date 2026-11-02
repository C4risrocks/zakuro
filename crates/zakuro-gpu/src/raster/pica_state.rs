//! Backend-independent PICA Draw block, matching raster.frag std140.
use crate::{fog, proctex};
use crate::lighting::Lighting;
use super::{BoundTexture, DepthMap, Wrap, TEXTURE_UNIT_BASES};

pub(crate) const DRAW_WORDS: usize = 408;

pub(crate) fn draw_uniform_block(
    words: &mut Vec<u32>, registers: &[u32], texture_config: u32,
    depth_flags: u32, depth_map: DepthMap, lighting: Option<&Lighting>,
) {
    words.clear();
    for base in [0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
        words.extend([registers[base], registers[base+1], registers[base+2], registers[base+3], registers[base+4], 0, 0, 0]);
    }
    words.extend([registers[0x0E0], registers[0x0FD], registers[0x0A0], texture_config]);
    for base in TEXTURE_UNIT_BASES {
        words.extend([registers[base+2], registers[base], 0, 0]);
    }
    words.extend([depth_flags, lighting.is_some() as u32, depth_map.scale.to_bits(), depth_map.offset.to_bits()]);
    match lighting {
        Some(lighting) => lighting.pack(words),
        None => words.resize(396, 0),
    }
    for register in proctex::REG_CONFIG..proctex::REG_CONFIG+6 { words.push(registers[register]); }
    words.extend([0, 0, registers[fog::REG_COLOR], 0, 0, 0]);
    debug_assert_eq!(words.len(), DRAW_WORDS);
}

#[cfg(test)]
mod tests {
    #[test]
    fn complete_draw_layout_and_three_texture_units() {
        let registers: Vec<u32> = (0..0x300).collect();
        let mut words = vec![u32::MAX; 600];
        super::draw_uniform_block(&mut words, &registers, 7, 3,
            super::DepthMap { scale: 0.5, offset: 0.25, w_buffer: true }, None);
        assert_eq!(words.len(), 408);
        assert_eq!(&words[48..52], &[0xE0, 0xFD, 0xA0, 7]);
        assert_eq!(&words[52..64], &[0x83, 0x81, 0, 0, 0x93, 0x91, 0, 0, 0x9B, 0x99, 0, 0]);
        assert_eq!(&words[64..68], &[3, 0, 0.5f32.to_bits(), 0.25f32.to_bits()]);
        assert!(words[68..396].iter().all(|word| *word == 0));
        assert_eq!(words[404], registers[crate::fog::REG_COLOR]);
    }
}

/// how a sampler filters and wraps, and goes between an image's sizes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Filtering {
    /// bilinear rather than point sampling, magnifying and minifying.
    pub(super) magnify: bool,
    pub(super) minify: bool,
    /// blends the two sizes about the level of detail rather than taking
    /// the nearer.
    pub(super) between: bool,
    pub(super) s: Wrap,
    pub(super) t: Wrap,
    /// the levels of detail it is kept between, see Lod, and its bias in
    /// 256ths.
    pub(super) levels: (u8, u8),
    pub(super) bias: i16,
}

impl Filtering {
    /// a unit's, for its texture's sizes.
    pub(super) fn of(bound: &BoundTexture) -> Filtering {
        let lod = bound.lod;
        match bound.levels {
            // one size is only ever magnified
            1 => Filtering { minify: bound.linear, between: false, levels: (0, 0), bias: 0, ..Filtering::wrapping(bound) },
            _ => Filtering { minify: lod.linear, between: lod.between, levels: (lod.min, lod.max), bias: lod.bias, ..Filtering::wrapping(bound) },
        }
    }

    /// for a texture pack's picture, going between its sizes down to
    /// smallest, a little sharper than the rate the coordinates change
    /// gives, so that one drawn a little smaller than its own size, as at 3
    /// times the console's resolution, stays as sharp as it is.
    pub(super) fn picture(bound: &BoundTexture, smallest: u32) -> Filtering {
        Filtering {
            minify: bound.linear,
            between: smallest > 0 && bound.linear,
            levels: (0, smallest as u8),
            bias: if smallest > 0 { -128 } else { 0 },
            ..Filtering::wrapping(bound)
        }
    }

    /// the unit's magnifying filter and wrapping, one size of its texture.
    pub(super) fn wrapping(bound: &BoundTexture) -> Filtering {
        Filtering { s: bound.wrap_s, t: bound.wrap_t, magnify: bound.linear, ..Filtering::OFF }
    }

    /// what a unit that is off samples with.
    pub(super) const OFF: Filtering =
        Filtering { magnify: false, minify: false, between: false, s: Wrap::ClampToEdge, t: Wrap::ClampToEdge, levels: (0, 0), bias: 0 };
}


/// Lighting, procedural and fog tables in the shader storage-buffer order.
pub(crate) fn table_words(out: &mut Vec<u32>, tables: &crate::lighting::Tables,
    procedural: &proctex::Tables, fog: &fog::Table) {
    out.clear();
    let (colors, steps) = procedural.colors();
    for part in [tables.entries().as_flattened().as_flattened(),
        procedural.maps().as_flattened().as_flattened(), colors.as_flattened(),
        steps.as_flattened(), fog.entries().as_flattened()] {
        out.extend(part.iter().map(|value| value.to_bits()));
    }
    debug_assert_eq!(out.len(), 15360);
}
