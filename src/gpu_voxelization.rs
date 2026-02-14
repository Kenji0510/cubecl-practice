use cubecl::prelude::*;

pub const EMPTY_KEY: u32 = 0xFFFF_FFFF;
pub const GRID_OFFSET_X: i32 = 512;
pub const GRID_OFFSET_Y: i32 = 512;
pub const GRID_OFFSET_Z: i32 = 512;

pub const P1: u32 = 73_856_093;
pub const P2: u32 = 19_349_663;
pub const P3: u32 = 83_492_791;

#[cube]
fn voxel_hash(px: f32, py: f32, pz: f32, voxel: f32) -> u32 {
    let inv = 1.0 / voxel;

    let vx = (px * inv).floor() as i32 + GRID_OFFSET_X;
    let vy = (py * inv).floor() as i32 + GRID_OFFSET_Y;
    let vz = (pz * inv).floor() as i32 + GRID_OFFSET_Z;

    let ux = vx as u32;
    let uy = vy as u32;
    let uz = vz as u32;

    (ux * P1) ^ (uy * P2) ^ (uz * P3)
}

#[cube(launch_unchecked)]
pub fn init_table(
    table_keys: &mut Array<u32>,
    table_counts: &mut Array<i32>,
    table_remap: &mut Array<i32>,
    table_centroids: &mut Array<f32>,
    table_size: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < table_size {
        table_keys[i as usize] = EMPTY_KEY;
        table_counts[i as usize] = 0;
        table_remap[i as usize] = -1;

        let b = (i as usize) * 3;
        table_centroids[b + 0] = 0.0;
        table_centroids[b + 1] = 0.0;
        table_centroids[b + 2] = 0.0;
    }
}

#[cube]
fn add_to_global(
    key: u32,
    px: f32,
    py: f32,
    pz: f32,
    count: i32,
    table_keys: &mut Array<Atomic<u32>>,
    table_centroids: &mut Array<Atomic<f32>>,
    table_counts: &mut Array<Atomic<i32>>,
    table_size: u32,
    #[comptime] global_probe: u32,
) {
    let mut idx = (key % table_size) as u32;

    let mut p: u32 = 0;
    let mut found = false;
    while p < global_probe && !found {
        let prev = table_keys[idx as usize].compare_exchange_weak(EMPTY_KEY, key);

        if prev == EMPTY_KEY || prev == key {
            let b = (idx as usize) * 3;
            table_centroids[b + 0].fetch_add(px);
            table_centroids[b + 1].fetch_add(py);
            table_centroids[b + 2].fetch_add(pz);
            table_counts[idx as usize].fetch_add(count);
            found = true;
        } else {
            idx += 1;
            if idx >= table_size {
                idx = 0;
            }
            p += 1;
        }
    }
}

#[cube(launch_unchecked)]
pub fn insert_points(
    points_xyz: &Array<f32>, // num_points*3
    num_points: u32,
    voxel: f32,

    table_keys: &mut Array<Atomic<u32>>,
    table_centroids: &mut Array<Atomic<f32>>, // table_size*3（sum）
    table_counts: &mut Array<Atomic<i32>>,
    table_size: u32,

    #[comptime] shared_table_size: u32, // 1536
    #[comptime] shared_probe: u32,      // 32
    #[comptime] global_probe: u32,      // 1000
) {
    // --- shared memory ---
    let s_keys = SharedMemory::<Atomic<u32>>::new(shared_table_size as usize);
    let s_cent = SharedMemory::<Atomic<f32>>::new((shared_table_size as usize) * 3);
    let s_cnt = SharedMemory::<Atomic<i32>>::new(shared_table_size as usize);

    let tid = UNIT_POS as u32;
    let ldim = CUBE_DIM as u32;

    let mut i: u32 = tid;
    while i < shared_table_size {
        s_keys[i as usize].store(EMPTY_KEY);
        s_cnt[i as usize].store(0);

        let b = (i as usize) * 3;
        s_cent[b + 0].store(0.0);
        s_cent[b + 1].store(0.0);
        s_cent[b + 2].store(0.0);

        i += ldim;
    }
    sync_cube();

    let pid = ABSOLUTE_POS as u32;
    if pid < num_points {
        let base = (pid * 3) as usize;
        let px = points_xyz[base + 0];
        let py = points_xyz[base + 1];
        let pz = points_xyz[base + 2];

        let key = voxel_hash(px, py, pz, voxel);
        let mut sidx = (key % shared_table_size) as u32;

        let mut stored: bool = false;
        let mut p: u32 = 0;
        while p < shared_probe {
            let prev = s_keys[sidx as usize].compare_exchange_weak(EMPTY_KEY, key);

            if prev == EMPTY_KEY || prev == key {
                let b = (sidx as usize) * 3;
                s_cent[b + 0].fetch_add(px);
                s_cent[b + 1].fetch_add(py);
                s_cent[b + 2].fetch_add(pz);
                s_cnt[sidx as usize].fetch_add(1);
                stored = true;
                break;
            }

            sidx += 1;
            if sidx >= shared_table_size {
                sidx = 0;
            }
            p += 1;
        }

        if !stored {
            add_to_global(
                key,
                px,
                py,
                pz,
                1,
                table_keys,
                table_centroids,
                table_counts,
                table_size,
                global_probe,
            );
        }
    }

    sync_cube();

    // --- shared flush ---
    let mut j: u32 = tid;
    while j < shared_table_size {
        let key = s_keys[j as usize].load();
        if key != EMPTY_KEY {
            let b = (j as usize) * 3;
            let sx = s_cent[b + 0].load();
            let sy = s_cent[b + 1].load();
            let sz = s_cent[b + 2].load();
            let sc = s_cnt[j as usize].load();

            add_to_global(
                key,
                sx,
                sy,
                sz,
                sc,
                table_keys,
                table_centroids,
                table_counts,
                table_size,
                global_probe,
            );
        }
        j += ldim;
    }
}

/// GLSL compact.glsl 相当（平均化 + out_count atomicAdd）
#[cube(launch_unchecked)]
pub fn compact(
    table_keys_raw: &Array<u32>,
    table_centroids_raw: &Array<f32>, // sum
    table_counts_raw: &Array<i32>,
    table_size: u32,

    out_points: &mut Array<f32>,    // max_out*3
    out_count: &mut Array<Atomic<u32>>, // 長さ1（out_count[0]）
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < table_size {
        let key = table_keys_raw[idx as usize];
        let cnt = table_counts_raw[idx as usize];

        if key != EMPTY_KEY && cnt > 0 {
            let b = (idx as usize) * 3;
            let mut sx = table_centroids_raw[b + 0];
            let mut sy = table_centroids_raw[b + 1];
            let mut sz = table_centroids_raw[b + 2];

            if cnt > 1 {
                let inv = 1.0 / (cnt as f32);
                sx *= inv;
                sy *= inv;
                sz *= inv;
            }

            let w = out_count[0].fetch_add(1) as usize;
            out_points[w * 3 + 0] = sx;
            out_points[w * 3 + 1] = sy;
            out_points[w * 3 + 2] = sz;
        }
    }
}
