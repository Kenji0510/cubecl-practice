use anyhow::{Context, Result};
use cubecl::{Runtime, prelude::*};
use cubecl_practice::{
    gpu_voxelization::{compact, init_table, insert_points},
    pcd::{PointXYZ, load_pcd_xyz, save_xyz_pcd},
};

struct PushConsts {
    num_points: i32,
    table_size: i32,
    voxel_size: f32,
    _pad: i32,
}

fn launch_voxelization<R: Runtime>(
    device: &R::Device,
    pts: &[f32],
    num_pts: usize,
    voxel_size: f32,
) -> Result<Vec<f32>> {
    let table_size = (num_pts * 4) as u32;
    let consts_data = PushConsts {
        num_points: num_pts as i32,
        table_size: table_size as i32,
        voxel_size,
        _pad: 0,
    };

    let client = R::client(device);

    let start = std::time::Instant::now();

    // Input points
    let pts_h = client.create_from_slice(f32::as_bytes(pts));

    // Create buffers
    let keys_h = client.empty((table_size as usize) * std::mem::size_of::<u32>());
    let cnt_h = client.empty((table_size as usize) * std::mem::size_of::<i32>());
    let remap_h = client.empty((table_size as usize) * std::mem::size_of::<i32>());
    let cent_h = client.empty((table_size as usize * 3) * std::mem::size_of::<f32>());

    let out_h = client.empty((table_size as usize * 3) * std::mem::size_of::<f32>());
    let outc_h = client.empty(1 * std::mem::size_of::<u32>());

    let (cc_init, cd_init) = launch_cfg(table_size);

    unsafe {
        init_table::launch_unchecked::<R>(
            &client,
            cc_init.clone(),
            cd_init.clone(),
            ArrayArg::from_raw_parts::<u32>(&keys_h, table_size as usize, 1),
            ArrayArg::from_raw_parts::<i32>(&cnt_h, table_size as usize, 1),
            ArrayArg::from_raw_parts::<i32>(&remap_h, table_size as usize, 1),
            ArrayArg::from_raw_parts::<f32>(&cent_h, table_size as usize * 3, 1),
            ScalarArg::new(table_size),
        )
        .unwrap();
    }

    let outc_h = client.create_from_slice(u32::as_bytes(&[0u32]));

    let shared_table_size: u32 = 1536;
    let shared_probe: u32 = 32;
    let block_size: u32 = 256;
    let global_probe: u32 = 1000;

    let (cc_ins,  cd_ins)  = launch_cfg(num_pts as u32);

    unsafe {
        insert_points::launch_unchecked::<R>(
            &client,
            cc_ins.clone(),
            cd_ins.clone(),
            ArrayArg::from_raw_parts::<f32>(&pts_h, (num_pts * 3) as usize, 1),
            ScalarArg::new(num_pts as u32),
            ScalarArg::new(voxel_size),
            ArrayArg::from_raw_parts::<u32>(&keys_h, table_size as usize, 1),
            ArrayArg::from_raw_parts::<f32>(&cent_h, table_size as usize * 3, 1),
            ArrayArg::from_raw_parts::<i32>(&cnt_h, table_size as usize, 1),
            ScalarArg::new(table_size),
            // shared_table_size,
            // shared_probe,
            block_size,
            global_probe,
        )
        .unwrap();
    }

    let (cc_cmp,  cd_cmp)  = launch_cfg(table_size);

    unsafe {
        compact::launch_unchecked::<R>(
            &client,
            cc_cmp.clone(),
            cd_cmp.clone(),
            ArrayArg::from_raw_parts::<u32>(&keys_h, table_size as usize, 1),
            ArrayArg::from_raw_parts::<f32>(&cent_h, (table_size as usize) * 3, 1),
            ArrayArg::from_raw_parts::<i32>(&cnt_h, table_size as usize, 1),
            ScalarArg::new(table_size),
            ArrayArg::from_raw_parts::<f32>(&out_h, (table_size as usize) * 3, 1),
            ArrayArg::from_raw_parts::<u32>(&outc_h, 1, 1),
        )
        .unwrap();
    }

    let outc_bytes = client.read_one(outc_h);
    let elapsed = start.elapsed();
    println!("Voxelization completed in {:.2?}", elapsed);

    let outc = u32::from_bytes(&outc_bytes)[0];
    println!("Output count: {}", outc);

    let out_bytes = client.read_one(out_h);
    let out_floats = f32::from_bytes(&out_bytes);
    Ok(out_floats[..(outc as usize * 3)].to_vec())
}

fn launch_cfg(n: u32) -> (CubeCount, CubeDim) {
    let block: u32 = 256;
    let grid = (n + block - 1) / block;
    (CubeCount::Static(grid, 1, 1), CubeDim::new_1d(block))
}


#[cfg(all(feature = "cuda", not(feature = "wgpu"), not(feature = "hip")))]
type R = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "hip", not(feature = "wgpu")))]
type R = cubecl::hip::HipRuntime;

#[cfg(feature = "wgpu")]
type R = cubecl::wgpu::WgpuRuntime;

fn backend_name() -> &'static str {
    #[cfg(all(feature = "cuda", not(feature = "wgpu"), not(feature = "hip")))]
    { "CUDA (NVIDIA)" }

    #[cfg(all(feature = "hip", not(feature = "wgpu")))]
    { "HIP (AMD)" }

    #[cfg(feature = "wgpu")]
    { "WGPU" }
}

fn main() -> Result<()> {
    println!("=== Checking available backends ===");
    println!("Using backend: {}\n", backend_name());

    let device = <R as Runtime>::Device::default();

    let pts_pcd = load_pcd_xyz("data/input/test/transformed-combined-frame-125.pcd")
        .context("Failed to load the pcd")?;

    let pts_vec = pcd_to_vec3f(&pts_pcd);
    let pts_f32: Vec<f32> = pts_vec
        .iter()
        .flat_map(|(x, y, z)| vec![*x, *y, *z])
        .collect();

    let voxel_size = 0.05;
    
    for i in   0..10 {
        let out = launch_voxelization::<R>(&device, &pts_f32, pts_vec.len(), voxel_size)
        .context("Failed to launch voxelization")?;
    }

    let out = launch_voxelization::<R>(&device, &pts_f32, pts_vec.len(), voxel_size)
        .context("Failed to launch voxelization")?;

    let downsampled_pcd = vec3f_to_pcd(&out);
    println!("Downsampled point count: {}", downsampled_pcd.len());

    save_xyz_pcd(&downsampled_pcd, "data/output/downsampled.pcd")
        .context("Failed to save the downsampled PCD")?;

    Ok(())
}

fn pcd_to_vec3f(pcd: &[PointXYZ]) -> Vec<(f32, f32, f32)> {
    pcd.iter()
        .map(|p| (p.x, p.y, p.z))
        .collect()
}

fn vec3f_to_pcd(pts_vec: &[f32]) -> Vec<PointXYZ> {
    pts_vec.chunks(3)
        .map(|chunk| PointXYZ {
            x: chunk[0],
            y: chunk[1],
            z: chunk[2],
        })
        .collect()
}