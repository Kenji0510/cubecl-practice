use cubecl::{Runtime, cube, prelude::*};


#[cube(launch_unchecked)]
fn vadd<F: Float>(a: &Array<Line<F>>, b: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

pub fn launch_vadd<R: Runtime>(device: &R::Device, a: &[f32], b: &[f32]) -> Vec<f32> {
    let vectorization = 4;

    let client = R::client(device);

    let a_handle = client.create_from_slice(f32::as_bytes(a));
    let b_handle = client.create_from_slice(f32::as_bytes(b));

    let out_handle = client.empty(a.len() * core::mem::size_of::<f32>());

    unsafe {
        vadd::launch_unchecked::<f32, R>(
            &client, 
            CubeCount::Static(1, 1, 1), 
            CubeDim::new_1d((a.len() / vectorization) as u32), 
            ArrayArg::from_raw_parts::<f32>(&a_handle, a.len(), vectorization as usize), 
            ArrayArg::from_raw_parts::<f32>(&b_handle, b.len(), vectorization as usize), 
            ArrayArg::from_raw_parts::<f32>(&out_handle, a.len(), vectorization as usize)
        )
        .unwrap();
    }

    let bytes = client.read_one(out_handle);
    f32::from_bytes(&bytes).to_vec()
}

fn main() {
    #[cfg(all(feature = "cuda", not(feature = "wgpu")))]
    type R = cubecl::cuda::CudaRuntime;
    
    #[cfg(feature = "wgpu")]
    type R = cubecl::wgpu::WgpuRuntime;

    let device = <R as Runtime>::Device::default();

    let a = vec![1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let b = vec![0.5_f32, 1.5, 2.5, 3.5, -1.0, -2.0, -3.0, -4.0];

    let out = launch_vadd::<R>(&device, &a, &b);

    println!("a:   {:?}", a);
    println!("b:   {:?}", b);
    println!("out: {:?}", out);
}
