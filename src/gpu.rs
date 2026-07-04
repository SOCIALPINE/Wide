//! Real GPU compute backend (wgpu, `gpu` feature — §4.3). The residency/transfer *model* has existed
//! since v0.8; this makes the computation itself real: matmul on `Device::Gpu` tensors dispatches an
//! actual WGSL compute shader. Buffers are cached on the tensor, so a chain of gpu-resident ops
//! re-uploads nothing (the §4.3 promise); each result is downloaded eagerly (autodiff/printing read
//! host data) and that D2H is illuminated honestly — lazy downloads are a later refinement.
//! If no adapter exists, everything falls back to the CPU path with an INFO (no hard failure).

use std::sync::OnceLock;

use wgpu::util::DeviceExt;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    matmul_pipeline: wgpu::ComputePipeline,
    ew_pipeline: wgpu::ComputePipeline,
    pub adapter_name: String,
}

static CTX: OnceLock<Option<Gpu>> = OnceLock::new();

/// The process-wide GPU context (device + queue + pipelines), created on first use.
/// None when no adapter is available — callers fall back to the CPU path.
pub fn ctx() -> Option<&'static Gpu> {
    CTX.get_or_init(init).as_ref()
}

const MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
struct Dims { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.m || col >= dims.n) { return; }
    var sum = 0.0;
    for (var i = 0u; i < dims.k; i = i + 1u) {
        sum = sum + a[row * dims.k + i] * b[i * dims.n + col];
    }
    c[row * dims.n + col] = sum;
}
"#;

const EW_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
struct P { op: u32, mode: u32, scalar: f32, len: u32 }
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    var x = a[i];
    var y = p.scalar;
    if (p.mode == 0u) { y = b[i]; }        // tensor OP tensor (same shape)
    if (p.mode == 2u) { let t = x; x = y; y = t; }  // scalar OP tensor (operand order)
    var r = 0.0;
    if (p.op == 0u) { r = x + y; }
    else if (p.op == 1u) { r = x - y; }
    else if (p.op == 2u) { r = x * y; }
    else { r = x / y; }                    // IEEE f32 division — same as the CPU tensor path
    c[i] = r;
}
"#;

fn init() -> Option<Gpu> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let adapter_name = adapter.get_info().name.clone();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wide-matmul"),
        source: wgpu::ShaderSource::Wgsl(MATMUL_WGSL.into()),
    });
    let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wide-matmul"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let ew_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wide-elementwise"),
        source: wgpu::ShaderSource::Wgsl(EW_WGSL.into()),
    });
    let ew_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wide-elementwise"),
        layout: None,
        module: &ew_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Some(Gpu { device, queue, matmul_pipeline, ew_pipeline, adapter_name })
}

fn as_bytes(data: &[f32]) -> &[u8] {
    // f32 slices are plain bytes (no padding) — a safe reinterpretation for buffer upload.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

/// Upload host data into a GPU storage buffer (H2D — the caller illuminates the byte count).
pub fn upload(g: &Gpu, data: &[f32]) -> wgpu::Buffer {
    g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wide-tensor"),
        contents: as_bytes(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

/// Run an elementwise op on the GPU (v0.51). `mode` 0 = tensor∘tensor (same length, `b` used),
/// 1 = tensor∘scalar, 2 = scalar∘tensor (`scalar` used; pass `b = a` to satisfy the layout).
/// `op` 0..3 = + - * /. Returns the result buffer (cached on the tensor) + downloaded data.
pub fn elementwise(g: &Gpu, a: &wgpu::Buffer, b: &wgpu::Buffer, op: u32, mode: u32, scalar: f32, len: usize) -> Result<(wgpu::Buffer, Vec<f32>), String> {
    let out_bytes = (len * 4) as u64;
    let c = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wide-ew-out"),
        size: out_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params: [u32; 4] = [op, mode, scalar.to_bits(), len as u32];
    let ubuf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wide-ew-params"),
        contents: unsafe { std::slice::from_raw_parts(params.as_ptr() as *const u8, 16) },
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let layout = g.ew_pipeline.get_bind_group_layout(0);
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ubuf.as_entire_binding() },
        ],
    });
    let staging = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wide-readback"),
        size: out_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&g.ew_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(len.div_ceil(64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&c, 0, &staging, 0, out_bytes);
    g.queue.submit(Some(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    g.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| "gpu readback channel closed".to_string())?
        .map_err(|e| format!("gpu readback failed: {:?}", e))?;
    let out = {
        let view = staging.slice(..).get_mapped_range();
        view.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<f32>>()
    };
    staging.unmap();
    Ok((c, out))
}

/// Run matmul (m,k)x(k,n) on the GPU. Returns the result's storage buffer (cached on the tensor so
/// chained ops re-upload nothing) and the downloaded host data (autodiff/printing need it).
pub fn matmul(g: &Gpu, a: &wgpu::Buffer, b: &wgpu::Buffer, m: usize, k: usize, n: usize) -> Result<(wgpu::Buffer, Vec<f32>), String> {
    let out_bytes = (m * n * 4) as u64;
    let c = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wide-matmul-out"),
        size: out_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dims: [u32; 4] = [m as u32, k as u32, n as u32, 0];
    let ubuf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wide-matmul-dims"),
        contents: unsafe { std::slice::from_raw_parts(dims.as_ptr() as *const u8, 16) },
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let layout = g.matmul_pipeline.get_bind_group_layout(0);
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ubuf.as_entire_binding() },
        ],
    });
    let staging = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wide-readback"),
        size: out_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&g.matmul_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n.div_ceil(8) as u32, m.div_ceil(8) as u32, 1);
    }
    enc.copy_buffer_to_buffer(&c, 0, &staging, 0, out_bytes);
    g.queue.submit(Some(enc.finish()));
    // synchronous readback (D2H) — the tree-walker is synchronous anyway
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    g.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| "gpu readback channel closed".to_string())?
        .map_err(|e| format!("gpu readback failed: {:?}", e))?;
    let out = {
        let view = staging.slice(..).get_mapped_range();
        view.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<f32>>()
    };
    staging.unmap();
    Ok((c, out))
}
