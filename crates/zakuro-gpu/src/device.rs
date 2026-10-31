//! one Vulkan device for the renderer and the presenter, when they would
//! pick the same GPU, so the screens the renderer draws are shown straight
//! from its images rather than copied out to the host and up again.

use ash::vk;

/// Portability drivers require their advertised subset extension at device creation.
pub(crate) fn device_extensions(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    extra: &[*const std::ffi::c_char],
) -> Result<Vec<*const std::ffi::c_char>, vk::Result> {
    let available = unsafe { instance.enumerate_device_extension_properties(physical) }?;
    let subset = c"VK_KHR_portability_subset";
    let mut extensions = extra.to_vec();
    if available.iter().any(|one| one.extension_name_as_c_str() == Ok(subset))
        && !extensions.iter().any(|one| unsafe { std::ffi::CStr::from_ptr(*one) } == subset)
    {
        extensions.push(subset.as_ptr());
    }
    Ok(extensions)
}

/// a device, the instance it came from and its one queue, gone once the
/// last of the renderer and the presenter lets go of it. both record and
/// submit on the thread the window runs on, so the queue needs no lock.
pub struct SharedDevice {
    pub(crate) _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) family: u32,
    pub(crate) device: ash::Device,
    pub(crate) queue: vk::Queue,
    /// whether the device does logic ops, and counts pipeline statistics.
    pub(crate) logic_ops: bool,
    pub(crate) statistics: bool,
    /// the blend state draws set as they go.
    pub(crate) dynamic: crate::raster::hardware::Dynamic,
}

impl SharedDevice {
    /// a device on the GPU the renderer would pick, with no presenter, to
    /// time showing the screens straight from the GPU without a window.
    pub fn new() -> Result<SharedDevice, String> {
        crate::raster::hardware::own_device()
    }
}

impl std::fmt::Debug for SharedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedDevice").field("family", &self.family).finish_non_exhaustive()
    }
}

impl Drop for SharedDevice {
    fn drop(&mut self) {
        // SAFETY: whoever made something from the device destroyed it before
        // letting go, the surface of the instance went with the presenter
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A submission completion whose fence can outlive the presenting window.
/// The shared device stays alive until the last image consumer releases it.
pub(crate) struct Submission {
    pub(crate) fence: vk::Fence,
    device: ash::Device,
    _owner: Option<std::sync::Arc<SharedDevice>>,
}

impl Submission {
    pub(crate) fn new(device: &ash::Device, owner: Option<std::sync::Arc<SharedDevice>>, signaled: bool) -> Result<Self, vk::Result> {
        let flags = if signaled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() };
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default().flags(flags), None) }?;
        Ok(Self { fence, device: device.clone(), _owner: owner })
    }
    pub(crate) fn ready(&self) -> Result<bool, vk::Result> {
        unsafe { self.device.get_fence_status(self.fence) }
    }
    pub(crate) fn wait(&self) -> Result<(), vk::Result> {
        unsafe { self.device.wait_for_fences(&[self.fence], true, u64::MAX) }
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        // A submitted fence is retained by the presenter or image consumers
        // until completion. An unsubmitted fence can be destroyed immediately.
        unsafe { self.device.destroy_fence(self.fence, None) };
    }
}

/// An upright output owned independently of its raster surface. Only these
/// images are pooled; the rasterizer's render targets remain private.
pub(crate) struct PresentationImage {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) width: u32,
    pub(crate) height: u32,
    memory: vk::DeviceMemory,
    pub(crate) owner: std::sync::Arc<SharedDevice>,
    consumers: std::sync::Mutex<Vec<std::sync::Arc<Submission>>>,
    pub(crate) generation: std::sync::atomic::AtomicU64,
}

impl PresentationImage {
    pub(crate) fn new(image: vk::Image, view: vk::ImageView, memory: vk::DeviceMemory, width: u32, height: u32, owner: std::sync::Arc<SharedDevice>) -> Self {
        Self { image, view, width, height, memory, owner, consumers: Default::default(), generation: Default::default() }
    }
    pub(crate) fn consumed_by(&self, submission: std::sync::Arc<Submission>) {
        let mut consumers = self.consumers.lock().unwrap();
        consumers.retain(|one| !matches!(one.ready(), Ok(true)));
        if !consumers.iter().any(|one| std::sync::Arc::ptr_eq(one, &submission)) {
            consumers.push(submission);
        }
    }
    pub(crate) fn ready(&self) -> Result<bool, vk::Result> {
        let mut consumers = self.consumers.lock().unwrap();
        let mut pending = false;
        for consumer in consumers.iter() { pending |= !consumer.ready()?; }
        if !pending { consumers.clear(); }
        Ok(!pending)
    }
    pub(crate) fn wait(&self) -> Result<(), vk::Result> {
        let mut consumers = self.consumers.lock().unwrap();
        for consumer in consumers.iter() { consumer.wait()?; }
        consumers.clear();
        Ok(())
    }
}

impl Drop for PresentationImage {
    fn drop(&mut self) {
        // Destruction is exceptional (surface eviction/resize/shutdown), never
        // the steady frame path. Also cover a producer not presented yet.
        unsafe {
            let _ = self.owner.device.device_wait_idle();
            self.owner.device.destroy_image_view(self.view, None);
            self.owner.device.destroy_image(self.image, None);
            self.owner.device.free_memory(self.memory, None);
        }
    }
}

impl std::fmt::Debug for PresentationImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PresentationImage").field("view", &self.view).finish_non_exhaustive()
    }
}

/// A bounded pool used by the producer and tested with the same completion
/// interface as the Vulkan adapter. A free entry never incurs a host wait.
pub(crate) struct FramePool<T> {
    entries: Vec<std::sync::Arc<T>>,
    next: usize,
}
impl<T> Default for FramePool<T> {
    fn default() -> Self { Self { entries: Vec::new(), next: 0 } }
}
impl<T> FramePool<T> {
    pub(crate) fn acquire<E>(
        &mut self,
        mut make: impl FnMut() -> Result<T, E>,
        ready: impl Fn(&T) -> Result<bool, E>,
        wait: impl Fn(&T) -> Result<(), E>,
    ) -> Result<std::sync::Arc<T>, E> {
        for (index, entry) in self.entries.iter().enumerate() {
            if ready(entry)? {
                self.next = (index + 1) % self.entries.len();
                return Ok(entry.clone());
            }
        }
        if self.entries.len() < 3 {
            let entry = std::sync::Arc::new(make()?);
            self.entries.push(entry.clone());
            self.next = 0;
            return Ok(entry);
        }
        let entry = self.entries[self.next].clone();
        wait(&entry)?;
        self.next = (self.next + 1) % self.entries.len();
        Ok(entry)
    }
}

#[cfg(test)]
mod frame_pool_tests {
    use super::FramePool;
    use std::cell::{Cell, RefCell};
    use std::sync::Arc;
    struct Image { number: usize, complete: Cell<bool> }
    #[test]
    fn frame_pool_waits_only_when_three_images_are_busy() {
        let mut pool = FramePool::default();
        let made = Cell::new(0);
        let waits = RefCell::new(Vec::new());
        let acquire = |pool: &mut FramePool<Image>| pool.acquire::<()>(
            || { let number = made.get(); made.set(number + 1); Ok(Image { number, complete: Cell::new(false) }) },
            |image| Ok(image.complete.get()),
            |image| { waits.borrow_mut().push(image.number); Ok(()) },
        ).unwrap();
        let first = acquire(&mut pool);
        let second = acquire(&mut pool);
        let third = acquire(&mut pool);
        assert_eq!(made.get(), 3);
        assert!(waits.borrow().is_empty());
        assert!(Arc::ptr_eq(&first, &acquire(&mut pool)));
        assert_eq!(*waits.borrow(), [0]);
        second.complete.set(true);
        assert!(Arc::ptr_eq(&second, &acquire(&mut pool)));
        assert_eq!(*waits.borrow(), [0]);
        assert_eq!(made.get(), 3);
        let weak = Arc::downgrade(&third);
        drop(pool);
        assert!(weak.upgrade().is_some(), "a consumer owns its image beyond the producer pool");
        drop(third);
        assert!(weak.upgrade().is_none());
    }
}

impl SharedDevice {
    /// Manufacture a GPU-only solid screen for the frontend's synthetic
    /// presentation diagnostic. No ROM, shader dump or player data is used.
    pub fn solid_screen(self: &std::sync::Arc<Self>, width: u32, height: u32, color: [f32; 4]) -> Result<crate::GpuScreen, String> {
        let device = &self.device;
        let map = |error: vk::Result| format!("synthetic GPU screen: {error:?}");
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let image = unsafe { device.create_image(&vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D).format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D { width, height, depth: 1 }).mip_levels(1).array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1).tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED), None) }.map_err(map)?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let properties = unsafe { self.instance.get_physical_device_memory_properties(self.physical) };
        let memory_type = (0..properties.memory_type_count).find(|index|
            requirements.memory_type_bits & (1 << index) != 0 && properties.memory_types[*index as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
        let Some(memory_type) = memory_type else {
            unsafe { device.destroy_image(image, None) };
            return Err("no memory type for synthetic GPU screen".into());
        };
        let memory = match unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(memory_type), None) } {
            Ok(memory) => memory,
            Err(error) => { unsafe { device.destroy_image(image, None) }; return Err(map(error)); }
        };
        let made = (|| {
            unsafe { device.bind_image_memory(image, memory, 0) }.map_err(map)?;
            unsafe { device.create_image_view(&vk::ImageViewCreateInfo::default().image(image)
                .view_type(vk::ImageViewType::TYPE_2D).format(vk::Format::R8G8B8A8_UNORM).subresource_range(range), None) }.map_err(map)
        })();
        let view = match made {
            Ok(view) => view,
            Err(error) => { unsafe { device.destroy_image(image, None); device.free_memory(memory, None); } return Err(error); }
        };
        let owned = std::sync::Arc::new(PresentationImage::new(image, view, memory, width, height, self.clone()));
        let pool = unsafe { device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(self.family), None) }.map_err(map)?;
        let produced = (|| {
            let command = unsafe { device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)) }.map_err(map)?[0];
            let fence = Submission::new(device, Some(self.clone()), false).map_err(map)?;
            unsafe {
                device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(map)?;
                let initial = vk::ImageMemoryBarrier::default().image(image).subresource_range(range)
                    .old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[initial]);
                device.cmd_clear_color_image(command, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: color }, &[range]);
                let ready = vk::ImageMemoryBarrier::default().image(image).subresource_range(range)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL).new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
                device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[ready]);
                device.end_command_buffer(command).map_err(map)?;
                device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&[command])], fence.fence).map_err(map)?;
            }
            fence.wait().map_err(map)
        })();
        unsafe { device.destroy_command_pool(pool, None) };
        produced?;
        Ok(crate::GpuScreen { image: owned, generation: 0, area: [0.0, 0.0, 1.0, 1.0], bounds: [0.0, 0.0, 1.0, 1.0] })
    }
}

#[cfg(test)]
mod gpu_readback_tests {
    use super::SharedDevice;
    use std::sync::Arc;

    #[test]
    #[ignore = "requires a Vulkan 1.3 loader and GPU; synthetic pixels only"]
    fn readback_keeps_the_selected_frame_and_crop() {
        let device = Arc::new(SharedDevice::new().expect("Vulkan device"));
        let normal = device.solid_screen(4, 4, [1.0, 0.0, 0.0, 1.0]).unwrap();
        // A distinct image stands for the interpolation twin; creating or
        // selecting another frame must not alter the captured frame handle.
        let mut selected_twin = device.solid_screen(4, 4, [0.0, 0.0, 1.0, 1.0]).unwrap();
        selected_twin.area = [0.25, 0.25, 0.5, 0.5];
        let (pixels, width, height) = selected_twin.read_pixels().unwrap();
        assert_eq!((width, height), (2, 2));
        assert_eq!(pixels, [0, 0, 255, 255].repeat(4));
        let (pixels, width, height) = normal.read_pixels().unwrap();
        assert_eq!((width, height), (4, 4));
        assert_eq!(pixels, [255, 0, 0, 255].repeat(16));
    }
}
