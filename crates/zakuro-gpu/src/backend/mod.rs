//! presentation backends.

#[cfg(feature = "opengl")]
pub mod gl;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "vulkan")]
mod vulkan_overlay;

/// one screen's pixels, already converted to straight RGBA8, or where they
/// are on the GPU, for a presenter sharing the renderer's device.
pub struct ScreenImage<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
    pub gpu: Option<GpuScreen>,
}

impl ScreenImage<'_> {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || (self.pixels.is_empty() && self.gpu.is_none())
    }
}

/// a screen's picture upright in an image of the renderer's, in the general
/// layout, which the Vulkan presenter draws straight from when the two share
/// a device.
#[derive(Debug, Clone)]
pub struct GpuScreen {
    #[cfg(feature = "vulkan")]
    pub(crate) image: std::sync::Arc<crate::device::PresentationImage>,
    #[cfg(feature = "vulkan")]
    pub(crate) generation: u64,
    /// where the screen's pixels lie in the image, as texture coordinates,
    /// the corner and the size, then the corners inset by half a texel,
    /// which filtering keeps within.
    pub(crate) area: [f32; 4],
    pub(crate) bounds: [f32; 4],
}

/// what is drawn over the screens, a user interface, as textured triangles
/// in window pixels.
#[derive(Default)]
pub struct Overlay {
    /// textures to make or change before drawing.
    pub textures: Vec<OverlayTexture>,
    pub meshes: Vec<OverlayMesh>,
    /// textures that go away once this frame is drawn.
    pub free: Vec<u64>,
}

/// pixels for a texture of the overlay.
pub struct OverlayTexture {
    pub id: u64,
    /// where the pixels go in a texture that exists, none to make it anew at
    /// this size.
    pub offset: Option<[u32; 2]>,
    pub size: [u32; 2],
    /// RGBA, premultiplied by alpha, in sRGB.
    pub pixels: Vec<u8>,
    /// filtered when scaled, rather than nearest.
    pub linear: bool,
}

/// one corner of an overlay triangle.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OverlayVertex {
    /// window pixels from the top left.
    pub position: [f32; 2],
    pub uv: [f32; 2],
    /// premultiplied sRGB.
    pub color: [u8; 4],
}

/// triangles of the overlay that share a texture and a clip.
pub struct OverlayMesh {
    pub texture: u64,
    /// the part of the window they may draw in, left, top, right, bottom.
    pub clip: [u32; 4],
    pub vertices: Vec<OverlayVertex>,
    pub indices: Vec<u32>,
}

/// where a screen goes in the window, in pixels from the top left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// how the screens are arranged in the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenLayout {
    /// the top screen over the bottom one, as on the console.
    #[default]
    Stacked,
    /// next to each other, the top screen on the left.
    SideBySide,
    /// the top screen alone.
    TopOnly,
    /// the bottom screen alone.
    BottomOnly,
    /// the top screen big, the bottom one small at its lower right.
    LargeTop,
}

/// how the screens' pixels are drawn when scaled to the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenFilter {
    /// blended between pixels.
    #[default]
    Smooth,
    /// whole pixels kept sharp and blended only where two meet, so that a
    /// screen scaled by an odd amount stays crisp without stairs.
    Sharp,
    /// every pixel a square of the nearest one.
    Pixels,
}

impl ScreenFilter {
    /// the number the present shaders know it by.
    pub(crate) fn mode(self) -> i32 {
        match self {
            ScreenFilter::Smooth => 0,
            ScreenFilter::Pixels => 1,
            ScreenFilter::Sharp => 2,
        }
    }
}

impl ScreenLayout {
    /// the size the screens take up together at the console's resolution,
    /// with the top screen top_width wide.
    pub fn size(self, top_width: f32) -> (u32, u32) {
        let (width, height) = self.size_f(top_width);
        (width.ceil() as u32, height as u32)
    }

    fn size_f(self, top_width: f32) -> (f32, f32) {
        match self {
            ScreenLayout::Stacked => (top_width.max(320.0), 480.0),
            ScreenLayout::SideBySide => (top_width + 320.0, 240.0),
            ScreenLayout::TopOnly => (top_width, 240.0),
            ScreenLayout::BottomOnly => (320.0, 240.0),
            ScreenLayout::LargeTop => (top_width + LARGE_TOP_BOTTOM.0 as f32, 240.0),
        }
    }
}

/// the top screen's width at the console's resolution, its own shape. a
/// wider one shows a game that a widescreen code makes draw a wider scene
/// into the same 400x240.
pub const TOP_WIDTH: f32 = 400.0;

/// the bottom screen's size next to the big top one, half the console's.
const LARGE_TOP_BOTTOM: (u32, u32) = (160, 120);

/// works out where the screens go inside a window, keeping the 3DS's
/// aspect ratio, or the top screen top_w wide, and centring them, scaled by
/// a whole number when integer asks and the window has room. a screen that
/// does not show has no place.
pub fn layout(window_width: u32, window_height: u32, arrangement: ScreenLayout, top_w: f32, integer: bool) -> (Option<Viewport>, Option<Viewport>) {
    let (total_width, total_height) = arrangement.size_f(top_w);

    let mut scale = (window_width as f32 / total_width).min(window_height as f32 / total_height);
    if integer && scale >= 1.0 {
        scale = scale.floor();
    }
    let mut offset_x = (window_width as f32 - total_width * scale) / 2.0;
    let mut offset_y = (window_height as f32 - total_height * scale) / 2.0;
    if integer {
        // whole pixels, or every texel would not get the same size
        offset_x = offset_x.floor();
        offset_y = offset_y.floor();
    }

    let top = (arrangement != ScreenLayout::BottomOnly).then_some(Viewport {
        x: offset_x,
        y: offset_y,
        width: top_w * scale,
        height: 240.0 * scale,
    });
    let bottom = match arrangement {
        // the bottom screen is narrower, so it is centered under the top one.
        ScreenLayout::Stacked => Some(Viewport {
            x: offset_x + (total_width - 320.0) / 2.0 * scale,
            y: offset_y + 240.0 * scale,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
        ScreenLayout::SideBySide => Some(Viewport {
            x: offset_x + top_w * scale,
            y: offset_y,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
        ScreenLayout::LargeTop => {
            // half the top screen's scale, a whole number of it when the
            // screens are scaled by whole numbers
            let half = (LARGE_TOP_BOTTOM.0 as f32 / 320.0) * scale;
            let small = if integer && half >= 1.0 { half.floor() } else { half };
            Some(Viewport {
                x: offset_x + top_w * scale,
                y: offset_y + 240.0 * scale - 240.0 * small,
                width: 320.0 * small,
                height: 240.0 * small,
            })
        }
        ScreenLayout::TopOnly => None,
        ScreenLayout::BottomOnly => Some(Viewport {
            x: offset_x,
            y: offset_y,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
    };
    (top, bottom)
}

/// how the top screen's two pictures, one for each eye, are shown when the
/// title draws in 3D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StereoMode {
    /// the left eye's picture alone, as with the 3D slider down.
    #[default]
    Off,
    /// the eyes mixed into red for the left one and cyan for the right, for
    /// glasses of those colors.
    Anaglyph,
    /// each eye in its half of the window, for 3D TVs and headsets.
    SideBySide,
    /// the same squeezed to half width, which a 3D TV stretches back.
    HalfSideBySide,
    /// the eyes on alternate rows of the window, for passive 3D monitors.
    Interlaced,
}

/// the 3D a presenter shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stereo {
    pub mode: StereoMode,
    /// the right eye's picture where the left one's goes, for viewing cross
    /// eyed, glasses worn the other way round or rows counted the other way.
    pub swap: bool,
}

/// which picture a place in the window shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shown {
    /// the top screen, as the left eye sees it.
    Top,
    /// the top screen as the right eye sees it.
    Right,
    Bottom,
    /// the top screen's two pictures mixed the way the stereo mode does.
    Both,
}

/// the screens a frame draws, where and from which picture. the 3D shows
/// when the title drew a picture for the right eye, side by side shows the
/// left one's twice otherwise, so a 3D TV keeps its halves.
pub fn place(window_width: u32, window_height: u32, arrangement: ScreenLayout, top_width: f32, integer: bool, stereo: Stereo, right: bool) -> Vec<(Viewport, Shown)> {
    let screens = |(top, bottom): (Option<Viewport>, Option<Viewport>), eye: Shown| {
        top.map(|viewport| (viewport, eye)).into_iter().chain(bottom.map(|viewport| (viewport, Shown::Bottom)))
    };
    let (first, second) = match (right, stereo.swap) {
        (false, _) => (Shown::Top, Shown::Top),
        (true, false) => (Shown::Top, Shown::Right),
        (true, true) => (Shown::Right, Shown::Top),
    };
    match stereo.mode {
        StereoMode::Off => screens(layout(window_width, window_height, arrangement, top_width, integer), Shown::Top).collect(),
        StereoMode::Anaglyph | StereoMode::Interlaced => {
            let top = if right { Shown::Both } else { Shown::Top };
            screens(layout(window_width, window_height, arrangement, top_width, integer), top).collect()
        }
        StereoMode::SideBySide => {
            // each half a window of its own
            let half = window_width / 2;
            let places = layout(half, window_height, arrangement, top_width, integer);
            let moved = |(viewport, shown): (Viewport, Shown)| (Viewport { x: viewport.x + half as f32, ..viewport }, shown);
            screens(places, first).chain(screens(places, second).map(moved)).collect()
        }
        StereoMode::HalfSideBySide => {
            // the whole window's places, squeezed into each half
            let places = layout(window_width, window_height, arrangement, top_width, integer);
            let half = window_width as f32 / 2.0;
            let squeezed = |by: f32| move |(viewport, shown): (Viewport, Shown)| {
                (Viewport { x: viewport.x / 2.0 + by, width: viewport.width / 2.0, ..viewport }, shown)
            };
            screens(places, first).map(squeezed(0.0)).chain(screens(places, second).map(squeezed(half))).collect()
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PresentError {
    #[error("{0}")]
    Backend(String),
    #[error("the swapchain is out of date and was recreated")]
    OutOfDate,
}

pub trait Presenter {
    fn name(&self) -> &'static str;

    /// draws both screens into the window, and the overlay over them. right
    /// is the top screen's picture for the right eye, empty when the title
    /// draws no 3D or none is shown.
    fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
        right: ScreenImage<'_>,
        overlay: &Overlay,
    ) -> Result<(), PresentError>;

    /// the window changed size.
    fn resize(&mut self, width: u32, height: u32);

    /// how the screens are arranged from the next frame on.
    fn set_layout(&mut self, arrangement: ScreenLayout);

    /// how wide the top screen shows, TOP_WIDTH for the console's shape.
    fn set_top_width(&mut self, width: f32);

    /// how the screens are filtered, and whether they are scaled by whole
    /// numbers, from the next frame on.
    fn set_scaling(&mut self, filter: ScreenFilter, integer: bool);

    /// how the 3D shows from the next frame on.
    fn set_stereo(&mut self, stereo: Stereo);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_keeps_the_aspect_ratio_and_centers() {
        // a window exactly 400x480 needs no scaling or offset.
        let (top, bottom) = layout(400, 480, ScreenLayout::Stacked, TOP_WIDTH, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!(top.x, 0.0);
        assert_eq!(top.y, 0.0);
        assert_eq!(top.width, 400.0);
        assert_eq!(bottom.y, 240.0);
        assert_eq!(bottom.x, 40.0);
        assert_eq!(bottom.width, 320.0);

        // doubling both dimensions doubles the scale.
        let top = layout(800, 960, ScreenLayout::Stacked, TOP_WIDTH, false).0.unwrap();
        assert_eq!(top.width, 800.0);
        assert_eq!(top.height, 480.0);

        // a window that is too wide letterboxes horizontally.
        let top = layout(1000, 480, ScreenLayout::Stacked, TOP_WIDTH, false).0.unwrap();
        assert_eq!(top.width, 400.0);
        assert_eq!(top.x, 300.0);
    }

    #[test]
    fn the_other_layouts_place_the_screens_their_way() {
        // side by side, the bottom screen starts where the top one ends
        let (top, bottom) = layout(1440, 480, ScreenLayout::SideBySide, TOP_WIDTH, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!((top.x, top.y, top.width), (0.0, 0.0, 800.0));
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (800.0, 0.0, 640.0, 480.0));

        // the top screen alone fills a window of its shape, and a taller
        // one centers it
        let (top, bottom) = layout(800, 960, ScreenLayout::TopOnly, TOP_WIDTH, false);
        assert!(bottom.is_none());
        let top = top.unwrap();
        assert_eq!((top.x, top.y, top.width, top.height), (0.0, 240.0, 800.0, 480.0));

        // and so does the bottom screen alone, a wider window centering it
        let (top, bottom) = layout(800, 480, ScreenLayout::BottomOnly, TOP_WIDTH, false);
        assert!(top.is_none());
        let bottom = bottom.unwrap();
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (80.0, 0.0, 640.0, 480.0));
    }

    #[test]
    fn the_large_top_screen_takes_most_of_the_window() {
        // the bottom screen at half size sits at the top one's lower right
        let (top, bottom) = layout(1120, 480, ScreenLayout::LargeTop, TOP_WIDTH, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!((top.x, top.y, top.width, top.height), (0.0, 0.0, 800.0, 480.0));
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (800.0, 240.0, 320.0, 240.0));
    }

    /// a wider top screen fills a window of its shape, and the screens
    /// around it move over to make room.
    #[test]
    fn a_wider_top_screen_takes_its_shape() {
        let wide = 240.0 * 16.0 / 9.0;
        let (top, _) = layout(1920, 1080, ScreenLayout::TopOnly, wide, false);
        let top = top.unwrap();
        assert_eq!((top.x, top.y, top.width.round(), top.height), (0.0, 0.0, 1920.0, 1080.0));

        // stacked, the bottom screen stays centered under it
        let (top, bottom) = layout(560, 480, ScreenLayout::Stacked, 560.0, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!((top.x, top.width), (0.0, 560.0));
        assert_eq!((bottom.x, bottom.width), (120.0, 320.0));

        // side by side, it starts where the top screen ends
        let (_, bottom) = layout(880, 240, ScreenLayout::SideBySide, 560.0, false);
        assert_eq!(bottom.unwrap().x, 560.0);
        assert_eq!(ScreenLayout::TopOnly.size(wide), (427, 240));
    }

    /// the screens each stereo mode draws, from which picture and where.
    #[test]
    fn the_stereo_modes_place_the_eyes() {
        let on = |mode| Stereo { mode, swap: false };
        let shown = |places: &[(Viewport, Shown)]| places.iter().map(|(_, shown)| *shown).collect::<Vec<_>>();
        let at = |places: &[(Viewport, Shown)], index: usize| {
            let viewport = places[index].0;
            (viewport.x, viewport.y, viewport.width, viewport.height)
        };

        // off, or a title drawing no 3D, the screens as ever
        let off = place(400, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::Off), true);
        assert_eq!(shown(&off), [Shown::Top, Shown::Bottom]);
        let flat = place(400, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::Anaglyph), false);
        assert_eq!(shown(&flat), [Shown::Top, Shown::Bottom]);

        // red and cyan or rows, both pictures where the top screen goes
        let mixed = place(400, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::Interlaced), true);
        assert_eq!(shown(&mixed), [Shown::Both, Shown::Bottom]);
        assert_eq!(at(&mixed, 0), (0.0, 0.0, 400.0, 240.0));

        // side by side, each half a window of its own
        let halves = place(800, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::SideBySide), true);
        assert_eq!(shown(&halves), [Shown::Top, Shown::Bottom, Shown::Right, Shown::Bottom]);
        assert_eq!(at(&halves, 0), (0.0, 0.0, 400.0, 240.0));
        assert_eq!(at(&halves, 2), (400.0, 0.0, 400.0, 240.0));
        assert_eq!(at(&halves, 3), (440.0, 240.0, 320.0, 240.0));

        // squeezed, the whole window's places at half width in each half
        let squeezed = place(800, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::HalfSideBySide), true);
        assert_eq!(shown(&squeezed), [Shown::Top, Shown::Bottom, Shown::Right, Shown::Bottom]);
        assert_eq!(at(&squeezed, 0), (100.0, 0.0, 200.0, 240.0));
        assert_eq!(at(&squeezed, 1), (120.0, 240.0, 160.0, 240.0));
        assert_eq!(at(&squeezed, 2), (500.0, 0.0, 200.0, 240.0));

        // swapped, the right eye on the left, and with no 3D drawn both
        // halves the left eye's
        let swapped = place(800, 480, ScreenLayout::Stacked, TOP_WIDTH, false, Stereo { mode: StereoMode::SideBySide, swap: true }, true);
        assert_eq!(shown(&swapped), [Shown::Right, Shown::Bottom, Shown::Top, Shown::Bottom]);
        let flat = place(800, 480, ScreenLayout::TopOnly, TOP_WIDTH, false, on(StereoMode::SideBySide), false);
        assert_eq!(shown(&flat), [Shown::Top, Shown::Top]);
    }

    #[test]
    fn integer_scaling_keeps_whole_multiples() {
        // 2.5 times fits, 2 is used, and the rest is border
        let top = layout(1000, 1200, ScreenLayout::Stacked, TOP_WIDTH, true).0.unwrap();
        assert_eq!((top.width, top.height), (800.0, 480.0));
        assert_eq!((top.x, top.y), (100.0, 120.0));
        // a window smaller than the console's screens still shows them whole
        let top = layout(200, 240, ScreenLayout::Stacked, TOP_WIDTH, true).0.unwrap();
        assert_eq!(top.width, 200.0);
        // an odd window puts the screens on whole pixels
        let (top, bottom) = layout(1001, 961, ScreenLayout::Stacked, TOP_WIDTH, true);
        assert_eq!((top.unwrap().x, top.unwrap().y), (100.0, 0.0));
        assert_eq!(bottom.unwrap().x, 180.0);
        // the small bottom screen of the big top layout keeps a whole scale
        let (top, bottom) = layout(1680, 720, ScreenLayout::LargeTop, TOP_WIDTH, true);
        assert_eq!(top.unwrap().width, 1200.0);
        assert_eq!((bottom.unwrap().width, bottom.unwrap().height), (320.0, 240.0));
    }
}

#[cfg(feature = "vulkan")]
impl GpuScreen {
    /// Recover this exact upright image/crop, including an interpolated twin.
    /// Used only by a presenter which cannot consume the shared Vulkan image.
    pub fn read_pixels(&self) -> Result<(Vec<u8>, u32, u32), String> {
        use ash::vk;
        let source = &self.image;
        if self.generation != source.generation.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("GPU frame is no longer current".into());
        }
        let [left, top, width, height] = self.area;
        let x = (left * source.width as f32).round() as u32;
        let y = (top * source.height as f32).round() as u32;
        let width = (width * source.width as f32).round() as u32;
        let height = (height * source.height as f32).round() as u32;
        if width == 0 || height == 0 || x.checked_add(width).is_none_or(|end| end > source.width)
            || y.checked_add(height).is_none_or(|end| end > source.height) {
            return Err("GPU frame crop exceeds its image".into());
        }
        let bytes = u64::from(width) * u64::from(height) * 4;
        let owner = source.owner.clone();
        let device = &owner.device;
        let map = |error: vk::Result| format!("GPU frame readback: {error:?}");
        // All exceptional paths retire submitted work before releasing buffers.
        struct Readback {
            owner: std::sync::Arc<crate::SharedDevice>,
            buffer: vk::Buffer,
            memory: vk::DeviceMemory,
            pool: vk::CommandPool,
        }
        impl Drop for Readback {
            fn drop(&mut self) {
                unsafe {
                    let _ = self.owner.device.device_wait_idle();
                    if self.pool != vk::CommandPool::null() { self.owner.device.destroy_command_pool(self.pool, None); }
                    self.owner.device.destroy_buffer(self.buffer, None);
                    if self.memory != vk::DeviceMemory::null() { self.owner.device.free_memory(self.memory, None); }
                }
            }
        }
        let buffer = unsafe { device.create_buffer(&vk::BufferCreateInfo::default().size(bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE), None) }.map_err(map)?;
        let mut held = Readback { owner: owner.clone(), buffer, memory: vk::DeviceMemory::null(), pool: vk::CommandPool::null() };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let properties = unsafe { owner.instance.get_physical_device_memory_properties(owner.physical) };
        let pick = |flags| (0..properties.memory_type_count).find(|index|
            requirements.memory_type_bits & (1 << index) != 0 && properties.memory_types[*index as usize].property_flags.contains(flags));
        let index = pick(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            .or_else(|| pick(vk::MemoryPropertyFlags::HOST_VISIBLE)).ok_or("no readable host memory for GPU frame")?;
        let coherent = properties.memory_types[index as usize].property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        held.memory = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(index), None) }.map_err(map)?;
        unsafe { device.bind_buffer_memory(buffer, held.memory, 0) }.map_err(map)?;
        held.pool = unsafe { device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(owner.family), None) }.map_err(map)?;
        let command = unsafe { device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(held.pool)
            .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)) }.map_err(map)?[0];
        let completion = crate::device::Submission::new(device, Some(owner.clone()), false).map_err(map)?;
        unsafe {
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(map)?;
            let ready = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[ready], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: 0 })
                .image_extent(vk::Extent3D { width, height, depth: 1 });
            device.cmd_copy_image_to_buffer(command, source.image, vk::ImageLayout::GENERAL, buffer, &[region]);
            let visible = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[visible], &[], &[]);
            device.end_command_buffer(command).map_err(map)?;
            device.queue_submit(owner.queue, &[vk::SubmitInfo::default().command_buffers(&[command])], completion.fence).map_err(map)?;
        }
        completion.wait().map_err(map)?;
        let address = unsafe { device.map_memory(held.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.map_err(map)?;
        if !coherent {
            if let Err(error) = unsafe { device.invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default().memory(held.memory).size(vk::WHOLE_SIZE)]) } {
                unsafe { device.unmap_memory(held.memory) };
                return Err(map(error));
            }
        }
        let pixels = unsafe { std::slice::from_raw_parts(address as *const u8, bytes as usize) }.to_vec();
        unsafe { device.unmap_memory(held.memory) };
        Ok((pixels, width, height))
    }
}
